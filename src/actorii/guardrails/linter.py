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
from typing import Any, Dict, List, Optional, Set, Tuple

from actorii.vm import CompiledMethod, Opcode


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
    "Net",
    "Web",
    "PDF",
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

    def collect_allowed_fields(
        self, class_name: str, receiver: Optional[Any] = None
    ) -> Set[str]:
        """Collect all declared `ivars` in `class_name` lineage plus live receiver fields."""
        allowed: Set[str] = set()
        if hasattr(self.vm, "_collect_class_ivars") and class_name:
            allowed.update(self.vm._collect_class_ivars(class_name))

        if receiver is not None:
            if hasattr(receiver, "_fields") and isinstance(receiver._fields, dict):
                allowed.update(str(k) for k in receiver._fields.keys())
            if hasattr(receiver, "_bridge_specs") and isinstance(receiver._bridge_specs, dict):
                allowed.update(str(k) for k in receiver._bridge_specs.keys())

        return allowed

    def _is_tier2_bridge_selector(self, selector: str) -> bool:
        """Check whether `selector` corresponds to a registered Tier 2 `@actor_command`."""
        hierarchy = getattr(self.vm, "hierarchy", None)
        if hierarchy is not None and hasattr(hierarchy, "_classes"):
            for cls in hierarchy._classes.values():
                specs = getattr(cls, "bridge_specs", None)
                if isinstance(specs, dict) and selector in specs:
                    if specs[selector].tier >= 2:
                        return True
        if isinstance(getattr(self.vm, "global_vars", None), dict):
            actor_dict = self.vm.global_vars.get("Actor", {})
            if isinstance(actor_dict, dict):
                for obj in actor_dict.values():
                    if hasattr(obj, "get_member_tier"):
                        tier = obj.get_member_tier(selector)
                        if tier is not None and tier >= 2:
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
        written_fields: List[str] = []
        called_selectors: List[str] = []
        globals_written: List[str] = []
        issues: List[LintIssue] = []

        # 1. Protected system class check & Ring 1 AgentHarness kernel guard
        if is_overwrite_check and self.protect_system_classes:
            if target_cls in PROTECTED_SYSTEM_CLASSES:
                if target_cls != "WebScraper" or getattr(self.vm, "lock_system_singletons", False):
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
                and getattr(self.vm, "lock_system_singletons", False)
            ):
                cls_obj = self.vm.hierarchy.get(target_cls) if hasattr(self.vm, "hierarchy") else None
                if cls_obj is not None and cls_obj.has_method(selector):
                    issues.append(
                        LintIssue(
                            code="SYS_PROTECTED_CLASS",
                            severity="error",
                            message=(
                                f"Cannot overwrite core Ring 1 kernel method 'AgentHarness>>{selector}'. "
                                f"Only cognitive hooks {sorted(RING1_MUTABLE_HARNESS_SELECTORS)} may be hot-patched."
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

        for idx_op, ((op, arg), consts) in enumerate(all_ops):
            const_val = consts[arg] if 0 <= arg < len(consts) else None
            if op == Opcode.GET_FIELD and isinstance(const_val, str):
                if const_val not in read_fields:
                    read_fields.append(const_val)
            elif op == Opcode.SET_FIELD and isinstance(const_val, str):
                if const_val not in written_fields:
                    written_fields.append(const_val)
            elif op == Opcode.STORE_GLOBAL and isinstance(const_val, str):
                if const_val not in globals_written:
                    globals_written.append(const_val)
            elif op == Opcode.SET_INDEX:
                has_index_mutation = True
                if "Actor" in consts:
                    has_global_Actor_mutation = True
                    if self.protect_system_classes and getattr(self.vm, "lock_system_singletons", False):
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
            elif op in (Opcode.CALL_METHOD, Opcode.CALL_EARLY, Opcode.CALL_GLOBAL):
                sel = _extract_method_call_name(const_val)
                if sel:
                    if sel not in called_selectors:
                        called_selectors.append(sel)
                    if self._is_tier2_bridge_selector(sel) or sel in (
                        "saveImage",
                        "loadImage",
                        "restoreImage",
                        "snapshot",
                    ):
                        has_tier2_call = True
                    if (
                        sel in ("atPut", "put", "add", "removeKey", "remove", "removeAt", "clear")
                        and "Actor" in consts
                        and self.protect_system_classes
                        and getattr(self.vm, "lock_system_singletons", False)
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
        allowed_fields = self.collect_allowed_fields(target_cls, receiver=receiver)
        if self.strict_ivars and allowed_fields:
            for rf in read_fields:
                if rf not in allowed_fields:
                    issues.append(
                        LintIssue(
                            code="LINT_UNDECLARED_IVAR",
                            severity="error",
                            message=(
                                f"Method '{target_cls}>>{selector}' reads undeclared field 'self.{rf}'. "
                                f"Allowed fields: {sorted(allowed_fields)}"
                            ),
                            class_name=target_cls,
                            selector=selector,
                        )
                    )
            for wf in written_fields:
                if wf not in allowed_fields:
                    issues.append(
                        LintIssue(
                            code="LINT_UNDECLARED_IVAR",
                            severity="error",
                            message=(
                                f"Method '{target_cls}>>{selector}' writes undeclared field 'self.{wf}'. "
                                f"Allowed fields: {sorted(allowed_fields)}"
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
        elif written_fields or has_index_mutation or "setField" in called_selectors:
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
