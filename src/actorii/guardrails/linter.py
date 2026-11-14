"""Static AST & Bytecode Linter and Risk-Tier Classifier for ActorII (`actorii.guardrails.linter`).

Inspects 1993 Actor `CompiledMethod` bytecode and AST nodes before execution or
class attachment to enforce:
1. Bytecode Risk Classification:
   - Tier 0 (`READ_ONLY`): Pure computation / field reads (`GET_FIELD` only).
   - Tier 1 (`LOCAL_MUTATION`): Mutates receiver fields (`SET_FIELD`) or collections (`SET_INDEX`).
   - Tier 2 (`SYSTEM_OR_BRIDGE`): Mutates globals (`STORE_GLOBAL`), modifies system classes,
     or invokes `@actor_command` Tier 2 Python bridge methods.
2. Instance Variable (`ivar`) Schema Validation:
   - Verifies that every `self.<field>` (`GET_FIELD` / `SET_FIELD`) matches a declared
     ivar in the class hierarchy (or an existing field on the live receiver).
3. Protected System Class Guard:
   - Prevents accidental or LLM-hallucinated overwrites of core 1993 class methods
     on `Object`, `Collection`, `Array`, `Dictionary`, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import threading
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from actorii.vm import CompiledMethod, Opcode

_LINTER_LOCK = threading.RLock()

_BUILTIN_TIER2_SELECTORS: FrozenSet[str] = frozenset(
    ("saveImage", "loadImage", "restoreImage", "snapshot")
)
_POOL_MUTATING_SELECTORS: FrozenSet[str] = frozenset(
    ("atPut", "put", "add", "removeKey", "remove", "removeAt", "clear")
)
_CALL_OPCODES: FrozenSet[Any] = frozenset(
    (Opcode.CALL_METHOD, Opcode.CALL_EARLY, Opcode.CALL_GLOBAL)
)
_MUTABLE_SCRAPER_CLASSES: FrozenSet[str] = frozenset(
    ("WebScraper", "ProductLearner")
)
_SORTED_RING1_HARNESS_SELECTORS: List[str] = [
    "buildStepPrompt",
    "instructionBlock",
    "invokeModel",
    "parseModelReply",
    "policyNote",
    "selectEngramContext",
    "selectGrammarContext",
    "transformAction",
    "verifyOutcome",
]


class RiskTier(IntEnum):
    """Risk classification for Actor bytecode chunks and methods."""

    TIER_0_READ_ONLY = 0
    TIER_1_LOCAL_MUTATION = 1
    TIER_2_SYSTEM_OR_BRIDGE = 2


PROTECTED_SYSTEM_CLASSES: Set[str] = {
    "Object",
    "Collection",
    "IndexedCollection",
    "Array",
    "OrderedCollection",
    "List",
    "Dictionary",
    "Interval",
    "Point",
    "Point3D",
    "Rect",
    "Set",
    "WindowsObject",
    "Window",
    "Dialog",
    "Control",
    "Button",
    "TextField",
    "Checkbox",
    "LLMBridge",
    "HostWorkspaceBridge",
    "VMReflectionBridge",
    "VMSandbox",
    "ActorGrammarGraph",
    "ActorMemoryGraph",
    "ActorBenchSuite",
    "ValkyrieSystemTwin",
    "SmartLightTwin",
    "PantryGraphTwin",
    "ThermoProTwin",
    "ThermometerUI",
    "ReverseEngineeringTwin",
    "ReverseProxyTwin",
    "JetsonEdgeTwin",
    "TelegramBotTwin",
    "NetTransportBridge",
    "PDFDocumentTwin",
    "WebScraper",
    "ProductLearner",
}

SYNTHESIZABLE_TWIN_CLASSES: Set[str] = {
    "ThermoProTwin",
    "ThermometerUI",
    "SmartLightTwin",
}

RING1_MUTABLE_HARNESS_SELECTORS: Set[str] = {
    "policyNote",
    "selectEngramContext",
    "selectGrammarContext",
    "instructionBlock",
    "buildStepPrompt",
    "invokeModel",
    "parseModelReply",
    "transformAction",
    "verifyOutcome",
}

PROTECTED_POOL_SINGLETONS: Set[str] = {
    "Sandbox",
    "Grammar",
    "Memory",
    "LLM",
    "VM",
    "Host",
    "Bench",
    "Hue",
    "Pantry",
    "System",
    "Valkyrie",
    "Harness",
    "Bilby",
    "ReverseEngineer",
    "REA",
    "Telegram",
    "Proxy",
    "Jetson",
    "Net",
    "Web",
    "PDF",
    "ProductLearner",
}


class ActorGuardrailError(RuntimeError):
    """Base exception raised when a guardrail policy blocks code compilation or execution."""


class ProtectedClassError(ActorGuardrailError):
    """Raised when attempting to overwrite a method on a protected system class."""


class OpcodeBudgetExceededError(ActorGuardrailError):
    """Raised when VM execution exceeds the configured `max_opcodes` gas budget."""


@dataclass
class LintIssue:
    """Single static analysis finding on an Actor method or bytecode chunk."""

    code: str
    severity: str  # "error" or "warning"
    message: str
    class_name: str = ""
    selector: str = ""


@dataclass
class LintReport:
    """Complete static analysis & risk classification report for a `CompiledMethod`."""

    class_name: str
    selector: str
    risk_tier: RiskTier
    issues: List[LintIssue] = field(default_factory=list)
    read_fields: List[str] = field(default_factory=list)
    written_fields: List[str] = field(default_factory=list)
    called_selectors: List[str] = field(default_factory=list)
    globals_written: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[LintIssue]:
        return [i for i in self.issues if i.severity == "error"]


def _extract_method_call_name(const_val: Any) -> str:
    if isinstance(const_val, dict):
        return str(const_val.get("method") or const_val.get("name") or "")
    if hasattr(const_val, "method"):
        return str(getattr(const_val, "method"))
    if hasattr(const_val, "name"):
        return str(getattr(const_val, "name"))
    if isinstance(const_val, str):
        return const_val
    return ""


def _walk_instructions_and_blocks(
    instructions: List[Tuple[Any, int]],
    constants: List[Any],
) -> List[Tuple[Tuple[Any, int], List[Any]]]:
    """Yield `(instruction, active_constants)` across a method and any nested blocks."""
    collected: List[Tuple[Tuple[Any, int], List[Any]]] = []
    for inst in instructions:
        collected.append((inst, constants))
        op, arg = inst
        if op == Opcode.MAKE_BLOCK and 0 <= arg < len(constants):
            block_desc = constants[arg]
            b_insts = getattr(block_desc, "instructions", None)
            b_consts = getattr(block_desc, "constants", None) or constants
            if b_insts:
                collected.extend(_walk_instructions_and_blocks(b_insts, b_consts))
    return collected


class ActorLinter:
    """Static bytecode and schema analyzer for ActorII."""

    def __init__(
        self,
        vm: Any,
        *,
        strict_ivars: bool = True,
        protect_system_classes: bool = False,
    ) -> None:
        self.vm = vm
        self.strict_ivars = strict_ivars
        self.protect_system_classes = protect_system_classes
        self._tier2_selector_cache: Optional[Set[str]] = None
        self._allowed_fields_cache: Dict[str, Set[str]] = {}

    def collect_allowed_fields(
        self, class_name: str, receiver: Optional[Any] = None
    ) -> Set[str]:
        """Collect all declared `ivars` in `class_name` lineage plus live receiver fields."""
        with _LINTER_LOCK:
            base_allowed = self._allowed_fields_cache.get(class_name)
            if base_allowed is None:
                base_allowed = set()
                if hasattr(self.vm, "_collect_class_ivars") and class_name:
                    base_allowed.update(self.vm._collect_class_ivars(class_name))
                self._allowed_fields_cache[class_name] = base_allowed

            if receiver is None:
                return set(base_allowed)

            allowed = set(base_allowed)
            if hasattr(receiver, "_fields") and isinstance(receiver._fields, dict):
                allowed.update(str(k) for k in list(receiver._fields.keys()))
            if hasattr(receiver, "_bridge_specs") and isinstance(receiver._bridge_specs, dict):
                allowed.update(str(k) for k in list(receiver._bridge_specs.keys()))

            return allowed

    def _get_tier2_bridge_selectors(self) -> Set[str]:
        """Build or return cached set of Tier 2 bridge command selectors on `self.vm`."""
        with _LINTER_LOCK:
            if self._tier2_selector_cache is not None:
                return self._tier2_selector_cache
            tier2: Set[str] = set(_BUILTIN_TIER2_SELECTORS)
            hierarchy = getattr(self.vm, "hierarchy", None)
            if hierarchy is not None and hasattr(hierarchy, "_classes"):
                for cls in list(hierarchy._classes.values()):
                    specs = getattr(cls, "bridge_specs", None)
                    if isinstance(specs, dict):
                        for sel, spec in list(specs.items()):
                            if getattr(spec, "tier", 0) >= 2:
                                tier2.add(str(sel))
            if isinstance(getattr(self.vm, "global_vars", None), dict):
                actor_dict = self.vm.global_vars.get("Actor", {})
                if isinstance(actor_dict, dict):
                    for obj in list(actor_dict.values()):
                        b_specs = getattr(obj, "_bridge_specs", None)
                        if isinstance(b_specs, dict):
                            for sel, spec in list(b_specs.items()):
                                if getattr(spec, "tier", 0) >= 2:
                                    tier2.add(str(sel))
            self._tier2_selector_cache = tier2
            return tier2

    def _is_tier2_bridge_selector(self, selector: str) -> bool:
        """Check whether `selector` corresponds to a registered Tier 2 `@actor_command`."""
        if selector in self._get_tier2_bridge_selectors():
            return True
        with _LINTER_LOCK:
            if isinstance(getattr(self.vm, "global_vars", None), dict):
                actor_dict = self.vm.global_vars.get("Actor", {})
                if isinstance(actor_dict, dict):
                    for obj in list(actor_dict.values()):
                        if hasattr(obj, "get_member_tier"):
                            tier = obj.get_member_tier(selector)
                            if tier is not None and tier >= 2:
                                if self._tier2_selector_cache is not None:
                                    self._tier2_selector_cache.add(selector)
                                return True
        return False

    def lint_method(
        self,
        compiled_method: CompiledMethod,
        *,
        class_name: Optional[str] = None,
        receiver: Optional[Any] = None,
        is_overwrite_check: bool = True,
    ) -> LintReport:
        """Analyze a `CompiledMethod` for ivar schema compliance, protected rings, and risk tier."""
        target_cls = class_name or compiled_method.class_name or "Object"
        selector = compiled_method.name

        read_fields: List[str] = []
        read_fields_set: Set[str] = set()
        written_fields: List[str] = []
        written_fields_set: Set[str] = set()
        called_selectors: List[str] = []
        called_selectors_set: Set[str] = set()
        globals_written: List[str] = []
        globals_written_set: Set[str] = set()
        issues: List[LintIssue] = []

        lock_singletons = bool(getattr(self.vm, "lock_system_singletons", False))

        # 1. Protected system class check & Ring 1 AgentHarness kernel guard
        if is_overwrite_check and self.protect_system_classes:
            if target_cls in PROTECTED_SYSTEM_CLASSES:
                if target_cls not in _MUTABLE_SCRAPER_CLASSES or lock_singletons:
                    cls_obj = self.vm.hierarchy.get(target_cls) if hasattr(self.vm, "hierarchy") else None
                    if cls_obj is not None and cls_obj.has_method(selector):
                        issues.append(
                            LintIssue(
                                code="SYS_PROTECTED_CLASS",
                                severity="error",
                                message=f"Cannot overwrite existing method '{selector}' on protected system class '{target_cls}'.",
                                class_name=target_cls,
                                selector=selector,
                            )
                        )
            elif (
                target_cls == "AgentHarness"
                and selector not in RING1_MUTABLE_HARNESS_SELECTORS
                and lock_singletons
            ):
                cls_obj = self.vm.hierarchy.get(target_cls) if hasattr(self.vm, "hierarchy") else None
                if cls_obj is not None and cls_obj.has_method(selector):
                    issues.append(
                        LintIssue(
                            code="SYS_PROTECTED_CLASS",
                            severity="error",
                            message=(
                                f"Cannot overwrite core Ring 1 kernel method 'AgentHarness>>{selector}'. "
                                f"Only cognitive hooks {_SORTED_RING1_HARNESS_SELECTORS} may be hot-patched."
                            ),
                            class_name=target_cls,
                            selector=selector,
                        )
                    )

        # 2. Walk bytecode instructions (including nested blocks)
        all_ops = _walk_instructions_and_blocks(
            compiled_method.instructions,
            compiled_method.constants,
        )

        has_index_mutation = False
        has_global_Actor_mutation = False
        has_tier2_call = False
        check_singleton_lock = self.protect_system_classes and lock_singletons

        for idx_op, ((op, arg), consts) in enumerate(all_ops):
            const_val = consts[arg] if 0 <= arg < len(consts) else None
            if op == Opcode.GET_FIELD and isinstance(const_val, str):
                if const_val not in read_fields_set:
                    read_fields_set.add(const_val)
                    read_fields.append(const_val)
            elif op == Opcode.SET_FIELD and isinstance(const_val, str):
                if const_val not in written_fields_set:
                    written_fields_set.add(const_val)
                    written_fields.append(const_val)
            elif op == Opcode.STORE_GLOBAL and isinstance(const_val, str):
                if const_val not in globals_written_set:
                    globals_written_set.add(const_val)
                    globals_written.append(const_val)
            elif op == Opcode.SET_INDEX:
                has_index_mutation = True
                if "Actor" in consts:
                    has_global_Actor_mutation = True
                    if check_singleton_lock:
                        # Inspect recent LOAD_CONST before SET_INDEX to detect Actor[#Singleton] := ...
                        for back_i in range(max(0, idx_op - 4), idx_op):
                            prev_op, prev_arg = all_ops[back_i][0]
                            prev_consts = all_ops[back_i][1]
                            if prev_op == Opcode.LOAD_CONST and 0 <= prev_arg < len(prev_consts):
                                sym_c = str(prev_consts[prev_arg]).lstrip("#")
                                if sym_c in PROTECTED_POOL_SINGLETONS:
                                    issues.append(
                                        LintIssue(
                                            code="SYS_PROTECTED_SINGLETON",
                                            severity="error",
                                            message=f"Cannot overwrite protected system singleton 'Actor[#{sym_c}]'.",
                                            class_name=target_cls,
                                            selector=selector,
                                        )
                                    )
                                    break
            elif op in _CALL_OPCODES:
                sel = _extract_method_call_name(const_val)
                if sel:
                    if sel not in called_selectors_set:
                        called_selectors_set.add(sel)
                        called_selectors.append(sel)
                    if not has_tier2_call and self._is_tier2_bridge_selector(sel):
                        has_tier2_call = True
                    if (
                        check_singleton_lock
                        and sel in _POOL_MUTATING_SELECTORS
                        and "Actor" in consts
                    ):
                        has_global_Actor_mutation = True
                        if sel == "clear":
                            issues.append(
                                LintIssue(
                                    code="SYS_PROTECTED_SINGLETON",
                                    severity="error",
                                    message="Cannot clear protected Actor[...] singleton pool.",
                                    class_name=target_cls,
                                    selector=selector,
                                )
                            )
                        else:
                            for back_i in range(max(0, idx_op - 4), idx_op):
                                prev_op, prev_arg = all_ops[back_i][0]
                                prev_consts = all_ops[back_i][1]
                                if prev_op == Opcode.LOAD_CONST and 0 <= prev_arg < len(prev_consts):
                                    sym_c = str(prev_consts[prev_arg]).lstrip("#")
                                    if sym_c in PROTECTED_POOL_SINGLETONS:
                                        issues.append(
                                            LintIssue(
                                                code="SYS_PROTECTED_SINGLETON",
                                                severity="error",
                                                message=f"Cannot mutate protected system singleton 'Actor[#{sym_c}]' via '{sel}'.",
                                                class_name=target_cls,
                                                selector=selector,
                                            )
                                        )
                                        break

        # 3. Validate declared instance variables (`ivars`) if class declares ivars
        if self.strict_ivars and (read_fields or written_fields):
            allowed_fields = self.collect_allowed_fields(target_cls, receiver=receiver)
            if allowed_fields:
                sorted_allowed: Optional[List[str]] = None
                for rf in read_fields:
                    if rf not in allowed_fields:
                        if sorted_allowed is None:
                            sorted_allowed = sorted(allowed_fields)
                        issues.append(
                            LintIssue(
                                code="LINT_UNDECLARED_IVAR",
                                severity="error",
                                message=(
                                    f"Method '{target_cls}>>{selector}' reads undeclared field 'self.{rf}'. "
                                    f"Allowed fields: {sorted_allowed}"
                                ),
                                class_name=target_cls,
                                selector=selector,
                            )
                        )
                for wf in written_fields:
                    if wf not in allowed_fields:
                        if sorted_allowed is None:
                            sorted_allowed = sorted(allowed_fields)
                        issues.append(
                            LintIssue(
                                code="LINT_UNDECLARED_IVAR",
                                severity="error",
                                message=(
                                    f"Method '{target_cls}>>{selector}' writes undeclared field 'self.{wf}'. "
                                    f"Allowed fields: {sorted_allowed}"
                                ),
                                class_name=target_cls,
                                selector=selector,
                            )
                        )

        # 4. Determine Risk Tier
        if (
            globals_written
            or has_global_Actor_mutation
            or has_tier2_call
            or (target_cls in PROTECTED_SYSTEM_CLASSES and target_cls not in SYNTHESIZABLE_TWIN_CLASSES)
        ):
            risk_tier = RiskTier.TIER_2_SYSTEM_OR_BRIDGE
        elif written_fields or has_index_mutation or "setField" in called_selectors_set:
            risk_tier = RiskTier.TIER_1_LOCAL_MUTATION
        else:
            risk_tier = RiskTier.TIER_0_READ_ONLY

        return LintReport(
            class_name=target_cls,
            selector=selector,
            risk_tier=risk_tier,
            issues=issues,
            read_fields=read_fields,
            written_fields=written_fields,
            called_selectors=called_selectors,
            globals_written=globals_written,
        )
