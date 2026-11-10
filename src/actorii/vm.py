# Virtual Machine for the 1993 Actor Language (clean-room reverse engineered)
# Implements a bytecode-based VM for executing actor language programs.
# Supports:
# - Object creation and destruction
# - Message passing / method dispatch
# - Single inheritance hierarchy
# - Instance fields and class methods
# - Full object lifecycle (init, run, cleanup)

"""Bytecode format (1993 Whitewater Group / Genesis Development Systems):
- 1-byte opcodes
- Variable-length operands (1-2 bytes for indices/immediates)
- Stack-based execution model
- Local variable frame
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import math
from typing import Any, Optional, Callable, Dict, Iterable, List, Tuple, Union

from actorii.core import (
    ActorObject,
    ActorClass,
    ActorInheritanceHierarchy,
    ActorMessage,
    Symbol,
)
from actorii.parser import BlockDescriptor, EarlyBoundDescriptor, CapturedVar


# Opcodes #####################################################################
class OpcodeValue(int):
    """Integer opcode that preserves its mnemonic in string representation."""

    def __new__(cls, val: int, name: str = ""):
        obj = super().__new__(cls, val)
        obj._name = name
        return obj

    def __str__(self) -> str:
        return self._name or super().__str__()

    def __repr__(self) -> str:
        return f"Opcode.{self._name}" if self._name else super().__repr__()


class Opcode:
    """1993 actor language opcodes (clean-room reverse engineered)."""

    # Stack operations & variable loads/stores
    PUSH_NULL = OpcodeValue(0x00, "PUSH_NULL")  # Push None onto stack
    PUSH_INT = OpcodeValue(0x01, "PUSH_INT")  # Push integer constant; operand = value
    PUSH_STR = OpcodeValue(
        0x02, "PUSH_STR"
    )  # Push string constant; operand = index into constant table
    PUSH_OBJ = OpcodeValue(
        0x03, "PUSH_OBJ"
    )  # Push object reference; operand = object index
    PUSH_GLOBAL = OpcodeValue(
        0x04, "PUSH_GLOBAL"
    )  # Push global variable; operand = index
    LOAD_GLOBAL = PUSH_GLOBAL  # Alias for PUSH_GLOBAL
    PUSH_LOCAL = OpcodeValue(
        0x05, "PUSH_LOCAL"
    )  # Push local variable; operand = offset
    LOAD_LOCAL = PUSH_LOCAL  # Alias for PUSH_LOCAL
    STORE_LOCAL = OpcodeValue(
        0x06, "STORE_LOCAL"
    )  # Pop top of stack into local variable slot; operand = slot
    STORE_GLOBAL = OpcodeValue(
        0x07, "STORE_GLOBAL"
    )  # Pop top of stack into global variable; operand = const index
    STORE_NAME = OpcodeValue(
        0x08, "STORE_NAME"
    )  # Pop top of stack into variable by name; operand = const index
    LOAD_NAME = OpcodeValue(
        0x09, "LOAD_NAME"
    )  # Load variable by name; operand = const index
    LOAD_CONST = OpcodeValue(
        0x0A, "LOAD_CONST"
    )  # Load constant from table; operand = const index

    # Object creation, calls & blocks
    NEW_OBJ = OpcodeValue(0x10, "NEW_OBJ")  # Create new instance; operand = class index
    NEW_ARRAY = OpcodeValue(
        0x11, "NEW_ARRAY"
    )  # Create Array collection; operand = size or 0
    CALL_METHOD = OpcodeValue(
        0x20, "CALL_METHOD"
    )  # Call a method; operand = method index / const index
    CALL_GLOBAL = OpcodeValue(
        0x21, "CALL_GLOBAL"
    )  # Call a global function; operand = index
    CALL_EARLY = OpcodeValue(
        0x22, "CALL_EARLY"
    )  # Early-bound call bypassing dynamic dispatch; operand = desc index
    INVOKE_PRIM = OpcodeValue(
        0x23, "INVOKE_PRIM"
    )  # Invoke built-in primitive (max, min, sqrt, asInt, dec, inc, fill)
    MAKE_BLOCK = OpcodeValue(
        0x24, "MAKE_BLOCK"
    )  # Create block closure; operand = block desc index
    CALL_BLOCK = OpcodeValue(
        0x25, "CALL_BLOCK"
    )  # Invoke block closure; operand = arg count

    # Instance fields & indexing
    GET_FIELD = OpcodeValue(
        0x30, "GET_FIELD"
    )  # Get instance field; operand = field name index
    SET_FIELD = OpcodeValue(
        0x31, "SET_FIELD"
    )  # Set instance field; operand = field name index
    GET_INDEX = OpcodeValue(
        0x32, "GET_INDEX"
    )  # Collection index read (coll[idx]); operand = 0
    SET_INDEX = OpcodeValue(
        0x33, "SET_INDEX"
    )  # Collection index write (coll[idx] := val); operand = 0

    # Inheritance & methods
    INHERIT = OpcodeValue(
        0x40, "INHERIT"
    )  # Establish inheritance; operand = parent class index
    DEF_METHOD = OpcodeValue(
        0x41, "DEF_METHOD"
    )  # Attach / define method to class; operand = const index

    # Binary operations & stack manipulation
    BINARY_ADD = OpcodeValue(0x50, "BINARY_ADD")  # Add top two stack values
    BINARY_SUB = OpcodeValue(
        0x51, "BINARY_SUB"
    )  # Subtract two stack values (left - right)
    BINARY_MUL = OpcodeValue(0x52, "BINARY_MUL")  # Multiply two stack values
    BINARY_DIV = OpcodeValue(0x53, "BINARY_DIV")  # Integer divide two stack values
    BINARY_EQ = OpcodeValue(0x54, "BINARY_EQ")  # Equal comparison (left == right)
    BINARY_NE = OpcodeValue(0x55, "BINARY_NE")  # Not equal comparison (left != right)
    POP = OpcodeValue(0x56, "POP")  # Pop top of stack
    POP_N = OpcodeValue(0x57, "POP_N")  # Pop N values; operand = count
    BINARY_LT = OpcodeValue(0x58, "BINARY_LT")  # Less than comparison (left < right)
    BINARY_LE = OpcodeValue(
        0x59, "BINARY_LE"
    )  # Less than or equal comparison (left <= right)
    BINARY_GT = OpcodeValue(0x5A, "BINARY_GT")  # Greater than comparison (left > right)
    BINARY_GE = OpcodeValue(
        0x5B, "BINARY_GE"
    )  # Greater than or equal comparison (left >= right)
    DUP = OpcodeValue(0x5C, "DUP")  # Duplicate top of stack
    BINARY_MOD = OpcodeValue(
        0x5D, "BINARY_MOD"
    )  # Modulo / remainder of two stack values (left % right)

    # Control flow
    JMP = OpcodeValue(0x60, "JMP")  # Unconditional jump; operand = offset
    JUMP_ABSOLUTE = JMP  # Alias for JMP
    JMP_IF_TRUE = OpcodeValue(
        0x61, "JMP_IF_TRUE"
    )  # Jump if top of stack is truthy; operand = offset
    JUMP_IF_TRUE = JMP_IF_TRUE  # Alias
    JMP_IF_FALSE = OpcodeValue(
        0x62, "JMP_IF_FALSE"
    )  # Jump if top of stack is falsy; operand = offset
    JUMP_IF_FALSE = JMP_IF_FALSE  # Alias for JMP_IF_FALSE
    JUMP_FORWARD = OpcodeValue(
        0x63, "JUMP_FORWARD"
    )  # Relative forward jump; operand = delta
    POP_JUMP_IF_FALSE = OpcodeValue(
        0x64, "POP_JUMP_IF_FALSE"
    )  # Pop and jump if falsy; operand = offset
    POP_JUMP_IF_TRUE = OpcodeValue(
        0x65, "POP_JUMP_IF_TRUE"
    )  # Pop and jump if truthy; operand = offset

    # Return & halt
    RETURN = OpcodeValue(0x70, "RETURN")  # Return from current function/method
    RETURN_VALUE = OpcodeValue(0x71, "RETURN_VALUE")  # Return value on top of stack
    RETURN_BLOCK = OpcodeValue(0x72, "RETURN_BLOCK")  # Return from block closure
    HALT = OpcodeValue(0xFF, "HALT")  # Halt VM execution


# Helper types ################################################################

Operand = int
Byte = int
Instruction = tuple[Opcode, Operand]


# =============================================================================
# Lexical Closures, Activation Frames & Standard 1993 Built-ins
# =============================================================================


class Cell:
    """A mutable storage cell for captured lexical variables across scopes."""

    __slots__ = ("value",)

    def __init__(self, value: Any = None):
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, val: Any) -> None:
        self.value = val

    def __repr__(self) -> str:
        return f"Cell({self.value!r})"


class BlockClosure:
    """Runtime representation of an instantiated block closure."""

    def __init__(
        self,
        descriptor: Any,
        outer_frame: Optional["StackFrame"] = None,
        home_frame: Optional["StackFrame"] = None,
        receiver: Any = None,
        captured_cells: Optional[Dict[int, Cell]] = None,
    ):
        self.descriptor = descriptor
        self.outer_frame = outer_frame
        self.home_frame = home_frame
        self.receiver = receiver
        self.captured_cells: Dict[int, Cell] = captured_cells or {}

    @property
    def params(self) -> List[str]:
        return getattr(self.descriptor, "params", [])

    @property
    def instructions(self) -> List[Tuple[Opcode, int]]:
        return getattr(self.descriptor, "instructions", [])

    @property
    def constants(self) -> List[Any]:
        return getattr(self.descriptor, "constants", [])

    @property
    def string_table(self) -> List[str]:
        return getattr(self.descriptor, "string_table", [])

    def __repr__(self) -> str:
        name = getattr(self.descriptor, "name", "Block")
        return f"<BlockClosure {name} params={self.params}>"


class NonLocalReturn(Exception):
    """Exception used to unwind Python and VM activation frames during non-local block return."""

    def __init__(self, value: Any, home_frame: Optional[Any] = None) -> None:
        super().__init__("NonLocalReturn")
        self.value = value
        self.home_frame = home_frame


class CompiledMethod:
    """Compiled bytecode method representation for ActorVM class hierarchy."""

    def __init__(
        self,
        name: str,
        class_name: str,
        params: List[str],
        locals: List[str],
        instructions: List[Tuple[Opcode, int]],
        constants: Optional[List[Any]] = None,
        string_table: Optional[List[str]] = None,
        receiver: str = "self",
        is_action: bool = False,
        source: Optional[str] = None,
        synthesized: bool = False,
    ) -> None:
        self.name: str = name
        self.class_name: str = class_name
        self.params: List[str] = params
        self.locals: List[str] = locals
        self.instructions: List[Tuple[Opcode, int]] = instructions
        self.constants: List[Any] = constants or []
        self.string_table: List[str] = string_table or []
        self.receiver: str = receiver
        self.is_action: bool = is_action
        self.source: Optional[str] = source
        self.synthesized: bool = synthesized

    @property
    def arity(self) -> int:
        return len(self.params)

    def __repr__(self) -> str:
        return (
            f"<CompiledMethod {self.class_name}>>{self.name}({', '.join(self.params)})>"
        )


@dataclass
class StackFrame:
    """Activation frame representing an execution context on the ActorVM call stack."""

    instructions: List[Tuple[Opcode, int]]
    pc: int = 0
    constants: List[Any] = field(default_factory=list)
    strings: List[str] = field(default_factory=list)
    local_vars: List[Any] = field(default_factory=list)
    receiver: Any = None
    caller_frame: Optional["StackFrame"] = None
    method_name: str = "<anonymous>"
    class_name: str = ""
    is_block: bool = False
    stack: List[Any] = field(default_factory=list)
    home_frame: Optional["StackFrame"] = None
    outer_frame: Optional["StackFrame"] = None
    closure: Optional[BlockClosure] = None
    is_active: bool = True

    def get_local(self, slot: int, default: Any = None) -> Any:
        if 0 <= slot < len(self.local_vars):
            val = self.local_vars[slot]
            if isinstance(val, Cell):
                return val.value
            return val
        return default

    def set_local(self, slot: int, value: Any) -> None:
        while len(self.local_vars) <= slot:
            self.local_vars.append(None)
        current = self.local_vars[slot]
        if isinstance(current, Cell):
            current.value = value
        else:
            self.local_vars[slot] = value

    @property
    def is_exhausted(self) -> bool:
        return self.pc >= len(self.instructions)

    def __repr__(self) -> str:
        kind = "Block" if self.is_block else "Frame"
        qual = f"{self.class_name}>>" if self.class_name else ""
        return (
            f"<{kind} {qual}{self.method_name} pc={self.pc}/{len(self.instructions)}>"
        )


from actorii.classes import (
    register_classes,
    ActorObject,
    ActorClass,
    ActorInheritanceHierarchy,
    ActorMessage,
    Symbol,
    ActorCollection,
    ActorIndexedCollection,
    ActorArray,
    ActorOrderedCollection,
    ActorList,
    ActorSet,
    Collection,
    IndexedCollection,
    Array,
    OrderedCollection,
    ListClass,
    Set,
    KeyedCollection,
    Dictionary,
    ActorDictionary,
    Interval,
    Point,
    Point3D,
    Rect,
    WindowsObject,
    Window,
    Dialog,
    Control,
    Button,
    TextField,
    Checkbox,
    Agent,
    ThermoProTwin,
    ThermometerUI,
)

# VM Core #####################################################################


class ActorVM:
    """Stack-based virtual machine for 1993 actor language bytecodes with multi-frame call stack."""

    _active_vm: Optional["ActorVM"] = None

    def __init__(self, hierarchy: Optional[ActorInheritanceHierarchy] = None) -> None:
        ActorVM._active_vm = self
        if hierarchy is None:
            hierarchy = ActorInheritanceHierarchy()
        self.hierarchy = hierarchy
        self.call_stack: List[StackFrame] = []
        self.global_vars: Dict[str, Any] = {}
        self.global_vars["Actor"] = ActorDictionary(self)
        self.halted: bool = False
        self.objects: List[ActorObject] = []
        self.next_obj_index: int = 0
        self._in_run_loop: bool = False
        self._current_compilation_class: str = ""
        self._last_popped: Any = None
        self.max_stack_depth: int = 500
        self.max_call_stack_depth: int = 500
        self.max_opcodes: Optional[int] = 500_000
        self._opcodes_executed: int = 0

        # Guardrails & Python-to-Actor Bridge state
        self.guardrails_enabled: bool = False
        self.strict_ivars: bool = True
        self.protect_system_classes: bool = False
        self.max_allowed_tier: int = 2
        self.allow_bridge_commands: bool = True
        self._in_shadow_run: bool = False
        self.fileout_dir: Optional[str] = None
        self.custom_judge: Optional[Callable[..., Any]] = None
        self.guardrail_audit_log: List[Dict[str, Any]] = []

        # AI + OOP Bridge & Dynamic JIT Method Synthesis state
        self.jit_synthesis_enabled: bool = False
        self.use_live_llm_for_jit: bool = False
        self.ai_provider: Optional[Callable[..., str]] = None
        self.ai_endpoint: str = "http://127.0.0.1:8080/v1/chat/completions"
        self.ai_model: str = "qwen3.8:latest"
        self.synthesized_methods: List[Dict[str, Any]] = []

        # Backward compatibility fallback storage when call_stack is empty
        self._instructions: List[Instruction] = []
        self._constants: List[Any] = []
        self._string_table: List[str] = []
        self._pc: int = 0
        self._stack: List[Any] = []
        self._local_vars: List[Any] = []

        self._init_handlers()
        self._init_builtins()

    def _init_builtins(self) -> None:
        """Initialize standard 1993 core classes in hierarchy and globals."""
        register_classes(self.hierarchy, self.global_vars["Actor"], self.global_vars)

    # =========================================================================
    # Transparent Frame Accessors (Backward Compatibility with existing tests)
    # =========================================================================

    @property
    def current_frame(self) -> Optional[StackFrame]:
        return self.call_stack[-1] if self.call_stack else None

    @property
    def stack(self) -> List[Any]:
        if self.call_stack:
            return self.call_stack[-1].stack
        return self._stack

    @stack.setter
    def stack(self, val: List[Any]) -> None:
        self._stack = val
        if self.call_stack:
            self.call_stack[-1].stack = val

    @property
    def local_vars(self) -> List[Any]:
        if self.call_stack:
            return self.call_stack[-1].local_vars
        return self._local_vars

    @local_vars.setter
    def local_vars(self, val: List[Any]) -> None:
        self._local_vars = val
        if self.call_stack:
            self.call_stack[-1].local_vars = val

    @property
    def pc(self) -> int:
        if self.call_stack:
            return self.call_stack[-1].pc
        return self._pc

    @pc.setter
    def pc(self, val: int) -> None:
        self._pc = val
        if self.call_stack:
            self.call_stack[-1].pc = val

    @property
    def instructions(self) -> List[Instruction]:
        if self.call_stack:
            return self.call_stack[-1].instructions
        return self._instructions

    @instructions.setter
    def instructions(self, val: List[Instruction]) -> None:
        self._instructions = val
        if self.call_stack:
            self.call_stack[-1].instructions = val

    @property
    def constants(self) -> List[Any]:
        if self.call_stack:
            return self.call_stack[-1].constants
        return self._constants

    @constants.setter
    def constants(self, val: List[Any]) -> None:
        self._constants = val
        if self.call_stack:
            self.call_stack[-1].constants = val

    @property
    def string_table(self) -> List[str]:
        if self.call_stack:
            return self.call_stack[-1].strings
        return self._string_table

    @string_table.setter
    def string_table(self, val: List[str]) -> None:
        self._string_table = val
        if self.call_stack:
            self.call_stack[-1].strings = val

    # =========================================================================
    # VM Lifecycle: Load & Run
    # =========================================================================

    def load(
        self,
        instructions: List[Instruction],
        constants: Optional[List[Any]] = None,
        string_table: Optional[List[str]] = None,
    ) -> None:
        """Load bytecode into the VM, initializing root module activation frame."""
        self._instructions = list(instructions)
        self._constants = list(constants) if constants is not None else []
        self._string_table = list(string_table) if string_table is not None else []
        self._pc = 0
        self._stack = []
        self._local_vars = []
        self.halted = False
        self.objects = []
        self.next_obj_index = 0

        root_frame = StackFrame(
            instructions=self._instructions,
            pc=0,
            constants=self._constants,
            strings=self._string_table,
            local_vars=self._local_vars,
            receiver=None,
            caller_frame=None,
            method_name="<module>",
            class_name="",
            is_block=False,
            stack=self._stack,
        )
        self.call_stack = [root_frame]

    def run(self) -> None:
        """Execute bytecode across multi-frame call stack until HALT or all frames return."""
        ActorVM._active_vm = self
        if not self._in_run_loop:
            self._opcodes_executed = 0
        self.halted = False
        self._in_run_loop = True

        if not self.call_stack and self._instructions:
            self.load(self._instructions, self._constants, self._string_table)

        initial_depth = len(self.call_stack)

        try:
            while self.call_stack and not self.halted:
                try:
                    while self.call_stack and not self.halted:
                        frame = self.call_stack[-1]
                        if frame.is_exhausted:
                            ret = (
                                frame.receiver
                                if (frame.receiver is not None and not frame.is_block)
                                else None
                            )
                            self._pop_frame(ret)
                            continue

                        opcode, operand = frame.instructions[frame.pc]
                        frame.pc += 1
                        self._execute_opcode(opcode, operand)
                except NonLocalReturn as nlr:
                    while self.call_stack and self.call_stack[-1] is not nlr.home_frame:
                        popped = self.call_stack.pop()
                        popped.is_active = False
                    if self.call_stack and self.call_stack[-1] is nlr.home_frame:
                        popped = self.call_stack.pop()
                        popped.is_active = False
                    if self.call_stack:
                        self.call_stack[-1].stack.append(nlr.value)
                    else:
                        self._stack.append(nlr.value)
                        self.halted = True
                        break
        except Exception:
            if self.call_stack and getattr(self, "_last_failed_frame_info", None) is None:
                top_f = self.call_stack[-1]
                cls_obj = self.hierarchy.get(top_f.class_name) if top_f.class_name else None
                m_obj = cls_obj.get_method(top_f.method_name) if cls_obj else None
                var_names = ["self"] + list(getattr(m_obj, "params", []) or []) + list(getattr(m_obj, "locals", []) or [])
                local_map = {}
                for idx_l, lv in enumerate(top_f.local_vars):
                    if idx_l == 0:
                        continue
                    v_name = var_names[idx_l] if idx_l < len(var_names) else f"local_{idx_l}"
                    raw_v = lv.value if isinstance(lv, Cell) else lv
                    local_map[str(v_name)] = repr(raw_v)[:120]
                self._last_failed_frame_info = {
                    "failedClass": str(top_f.class_name or "<Chunk>"),
                    "failedSelector": str(top_f.method_name or "<chunk>"),
                    "bytecodeOffset": max(0, int(top_f.pc) - 1),
                    "localBindings": local_map,
                }
            while len(self.call_stack) > initial_depth:
                popped = self.call_stack.pop()
                popped.is_active = False
            if self.call_stack:
                self.call_stack[-1].is_active = False
            raise
        finally:
            self._in_run_loop = False
            if not self.call_stack:
                self.halted = True

    def _push_frame(self, frame: StackFrame) -> None:
        """Push a new activation frame onto the call stack."""
        if len(self.call_stack) >= self.max_stack_depth:
            raise RuntimeError("StackOverflowError: Call stack depth exceeded limit")
        self.call_stack.append(frame)

    def _pop_frame(self, return_val: Any = None) -> Optional[StackFrame]:
        """Pop the active frame and restore caller frame context."""
        if not self.call_stack:
            self.halted = True
            return None

        finished = self.call_stack.pop()
        finished.is_active = False

        if self.call_stack:
            caller = self.call_stack[-1]
            if return_val is not None:
                caller.stack.append(return_val)
            return caller
        else:
            self.halted = True
            self._instructions = finished.instructions
            self._constants = finished.constants
            self._string_table = finished.strings
            self._pc = finished.pc
            self._local_vars = finished.local_vars
            self._stack = finished.stack
            if return_val is not None:
                self._stack.append(return_val)
            return None

    # =========================================================================
    # Opcode Dispatch & Handlers
    # =========================================================================

    def _init_handlers(self) -> None:
        self.handlers = {
            Opcode.PUSH_NULL: self._handle_push_null,
            Opcode.PUSH_INT: self._handle_push_int,
            Opcode.PUSH_STR: self._handle_push_str,
            Opcode.PUSH_OBJ: self._handle_push_obj,
            Opcode.PUSH_GLOBAL: self._handle_push_global,
            Opcode.PUSH_LOCAL: self._handle_push_local,
            Opcode.STORE_LOCAL: self._handle_store_local,
            Opcode.STORE_GLOBAL: self._handle_store_global,
            Opcode.STORE_NAME: self._handle_store_name,
            Opcode.LOAD_NAME: self._handle_load_name,
            Opcode.LOAD_CONST: self._handle_load_const,
            Opcode.NEW_OBJ: self._handle_new_obj,
            Opcode.NEW_ARRAY: self._handle_new_array,
            Opcode.CALL_METHOD: self._handle_call_method,
            Opcode.CALL_GLOBAL: self._handle_call_global,
            Opcode.CALL_EARLY: self._handle_call_early,
            Opcode.INVOKE_PRIM: self._handle_invoke_prim,
            Opcode.MAKE_BLOCK: self._handle_make_block,
            Opcode.CALL_BLOCK: self._handle_call_block,
            Opcode.GET_FIELD: self._handle_get_field,
            Opcode.SET_FIELD: self._handle_set_field,
            Opcode.GET_INDEX: self._handle_get_index,
            Opcode.SET_INDEX: self._handle_set_index,
            Opcode.INHERIT: self._handle_inherit,
            Opcode.DEF_METHOD: self._handle_def_method,
            Opcode.BINARY_ADD: self._handle_binary_add,
            Opcode.BINARY_SUB: self._handle_binary_sub,
            Opcode.BINARY_MUL: self._handle_binary_mul,
            Opcode.BINARY_DIV: self._handle_binary_div,
            Opcode.BINARY_MOD: self._handle_binary_mod,
            Opcode.BINARY_EQ: self._handle_binary_eq,
            Opcode.BINARY_NE: self._handle_binary_ne,
            Opcode.POP: self._handle_pop,
            Opcode.POP_N: self._handle_pop_n,
            Opcode.BINARY_LT: self._handle_binary_lt,
            Opcode.BINARY_LE: self._handle_binary_le,
            Opcode.BINARY_GT: self._handle_binary_gt,
            Opcode.BINARY_GE: self._handle_binary_ge,
            Opcode.DUP: self._handle_dup,
            Opcode.JMP: self._handle_jmp,
            Opcode.JMP_IF_TRUE: self._handle_jmp_if_true,
            Opcode.JMP_IF_FALSE: self._handle_jmp_if_false,
            Opcode.JUMP_FORWARD: self._handle_jump_forward,
            Opcode.POP_JUMP_IF_FALSE: self._handle_pop_jump_if_false,
            Opcode.POP_JUMP_IF_TRUE: self._handle_pop_jump_if_true,
            Opcode.RETURN: self._handle_return,
            Opcode.RETURN_VALUE: self._handle_return_value,
            Opcode.RETURN_BLOCK: self._handle_return_block,
            Opcode.HALT: self._handle_halt,
        }

    def _execute_opcode(self, opcode: Opcode, operand: Operand) -> None:
        """Dispatch to the opcode handler."""
        self._opcodes_executed += 1
        if self.max_opcodes is not None and self._opcodes_executed > self.max_opcodes:
            from actorii.guardrails.linter import OpcodeBudgetExceededError

            raise OpcodeBudgetExceededError(
                f"OpcodeBudgetExceededError: Execution exceeded max_opcodes budget ({self.max_opcodes})"
            )
        handler = self.handlers.get(opcode)
        if handler is None:
            raise ValueError(f"Unknown opcode: 0x{opcode:02x}")
        handler(operand)

    def _handle_push_null(self, operand: Operand) -> None:
        self.stack.append(None)

    def _handle_push_int(self, value: Operand) -> None:
        self.stack.append(value)

    def _handle_push_str(self, str_idx: Operand) -> None:
        if self.string_table and str_idx < len(self.string_table):
            self.stack.append(self.string_table[str_idx])
        else:
            self.stack.append("")

    def _handle_push_obj(self, obj_idx: Operand) -> None:
        if obj_idx < len(self.objects):
            self.stack.append(self.objects[obj_idx])
        else:
            self.stack.append(None)

    def _handle_push_global(self, global_idx: Operand) -> None:
        if global_idx < len(self.constants):
            name = str(self.constants[global_idx])
            val = None
            if name in self.global_vars:
                val = self.global_vars[name]
            elif "Actor" in self.global_vars and name in self.global_vars["Actor"]:
                val = self.global_vars["Actor"][name]
            elif hasattr(self.hierarchy, "get"):
                val = self.hierarchy.get(name)
            self.stack.append(val)
        else:
            self.stack.append(None)

    def _handle_push_local(self, local_idx: Operand) -> None:
        if local_idx < len(self.local_vars):
            val = self.local_vars[local_idx]
            if isinstance(val, Cell):
                val = val.value
            self.stack.append(val)
        else:
            self.stack.append(None)

    def _handle_store_local(self, local_idx: Operand) -> None:
        if not self.stack:
            return
        val = self.stack.pop()
        self._last_popped = val
        while len(self.local_vars) <= local_idx:
            self.local_vars.append(None)
        if isinstance(self.local_vars[local_idx], Cell):
            self.local_vars[local_idx].value = val
        else:
            self.local_vars[local_idx] = val

    def _handle_store_global(self, global_idx: Operand) -> None:
        if not self.stack:
            return
        val = self.stack.pop()
        self._last_popped = val
        if global_idx < len(self.constants):
            name = str(self.constants[global_idx])
            norm = name[1:] if name.startswith("#") else name
            if (
                self.protect_system_classes
                and getattr(self, "lock_system_singletons", False)
            ):
                from actorii.guardrails.linter import (
                    PROTECTED_POOL_SINGLETONS,
                    ProtectedClassError,
                )

                if norm in PROTECTED_POOL_SINGLETONS and (
                    norm in self.global_vars
                    or norm in self.global_vars.get("Actor", {})
                ):
                    raise ProtectedClassError(
                        f"ProtectedSingletonError: Cannot overwrite locked system singleton 'Actor[#{norm}]'"
                    )
            self.global_vars[name] = val
            if "Actor" in self.global_vars and isinstance(
                self.global_vars["Actor"], dict
            ):
                self.global_vars["Actor"][name] = val

    def _handle_store_name(self, name_idx: Operand) -> None:
        if not self.stack:
            return
        val = self.stack.pop()
        self._last_popped = val
        if name_idx < len(self.constants):
            name = str(self.constants[name_idx])
            norm = name[1:] if name.startswith("#") else name
            if (
                self.protect_system_classes
                and getattr(self, "lock_system_singletons", False)
            ):
                from actorii.guardrails.linter import (
                    PROTECTED_POOL_SINGLETONS,
                    ProtectedClassError,
                )

                if norm in PROTECTED_POOL_SINGLETONS and (
                    norm in self.global_vars
                    or norm in self.global_vars.get("Actor", {})
                ):
                    raise ProtectedClassError(
                        f"ProtectedSingletonError: Cannot overwrite locked system singleton 'Actor[#{norm}]'"
                    )
            self.global_vars[name] = val
            if "Actor" in self.global_vars and isinstance(
                self.global_vars["Actor"], dict
            ):
                self.global_vars["Actor"][name] = val

    def _handle_load_name(self, name_idx: Operand) -> None:
        if name_idx < len(self.constants):
            name = str(self.constants[name_idx])
            if name in self.global_vars:
                self.stack.append(self.global_vars[name])
            elif "Actor" in self.global_vars and name in self.global_vars["Actor"]:
                self.stack.append(self.global_vars["Actor"][name])
            elif hasattr(self.hierarchy, "get"):
                self.stack.append(self.hierarchy.get(name))
            else:
                self.stack.append(None)
        else:
            self.stack.append(None)

    def _handle_load_const(self, const_idx: Operand) -> None:
        if const_idx < len(self.constants):
            val = self.constants[const_idx]
            if (
                isinstance(val, list)
                and val
                and type(val[0]).__module__.startswith("actorii.parser")
            ):
                n = len(val)
                if len(self.stack) >= n:
                    items = [self.stack.pop() for _ in range(n)][::-1]
                    self.stack.append(items)
                    return
            self.stack.append(val)
        else:
            self.stack.append(None)

    def _handle_new_obj(self, class_idx: Operand) -> None:
        obj = ActorObject()
        obj.class_idx = class_idx
        obj.methods = {}
        obj.fields = {}
        self.objects.append(obj)
        self.next_obj_index += 1
        self.stack.append(self.next_obj_index - 1)

    def _handle_new_array(self, operand: Operand) -> None:
        size = self.stack.pop() if self.stack and operand != 0 else operand
        self.stack.append([None] * (int(size) if size else 0))

    def _handle_get_field(self, field_idx: Operand) -> None:
        if self.stack:
            obj = self.stack.pop()
            field_name = None
            if isinstance(field_idx, str):
                field_name = field_idx
            elif isinstance(field_idx, int) and field_idx < len(self.constants):
                cand = self.constants[field_idx]
                if isinstance(cand, str):
                    field_name = cand
                elif hasattr(cand, "name"):
                    field_name = getattr(cand, "name")

            val = None
            if obj is not None:
                if hasattr(obj, "_fields") and isinstance(obj._fields, dict):
                    if field_name and field_name in obj._fields:
                        val = obj._fields[field_name]
                    elif isinstance(field_idx, int) and field_idx < len(obj._fields):
                        val = list(obj._fields.values())[field_idx]
                elif hasattr(obj, "fields") and isinstance(obj.fields, dict):
                    if field_name and field_name in obj.fields:
                        val = obj.fields[field_name]
                    elif isinstance(field_idx, int) and field_idx < len(obj.fields):
                        val = list(obj.fields.values())[field_idx]
                elif field_name and hasattr(obj, field_name):
                    val = getattr(obj, field_name)

            self.stack.append(val)
        else:
            self.stack.append(None)

    def _handle_set_field(self, field_idx: Operand) -> None:
        if len(self.stack) >= 2:
            value = self.stack.pop()
            obj = self.stack.pop()
            field_name = None
            if isinstance(field_idx, str):
                field_name = field_idx
            elif isinstance(field_idx, int) and field_idx < len(self.constants):
                cand = self.constants[field_idx]
                if isinstance(cand, str):
                    field_name = cand
                elif hasattr(cand, "name"):
                    field_name = getattr(cand, "name")

            if obj is not None:
                if hasattr(obj, "_fields") and isinstance(obj._fields, dict):
                    if field_name:
                        obj._fields[field_name] = value
                    elif isinstance(field_idx, int) and field_idx < len(obj._fields):
                        key = list(obj._fields.keys())[field_idx]
                        obj._fields[key] = value
                    else:
                        obj._fields[str(field_idx)] = value
                elif hasattr(obj, "fields") and isinstance(obj.fields, dict):
                    if field_name:
                        obj.fields[field_name] = value
                    elif isinstance(field_idx, int) and field_idx < len(obj.fields):
                        key = list(obj.fields.keys())[field_idx]
                        obj.fields[key] = value
                    else:
                        obj.fields[str(field_idx)] = value
                elif field_name:
                    try:
                        setattr(obj, field_name, value)
                    except Exception:
                        pass

            self.stack.append(value)

    def _handle_inherit(self, parent_idx: Operand) -> None:
        if self.stack:
            obj = self.stack[-1]
            if obj is not None and hasattr(obj, "parent"):
                obj.parent = parent_idx

    def _handle_def_method(self, operand: Operand) -> None:
        pass

    # =========================================================================
    # Arithmetic & Binary Operations
    # =========================================================================

    def _handle_binary_add(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0 if not isinstance(right, str) else ""
        if right is None:
            right = 0 if not isinstance(left, str) else ""
        self.stack.append(left + right)

    def _handle_binary_sub(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 0
        self.stack.append(left - right)

    def _handle_binary_mul(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 0
        self.stack.append(left * right)

    def _handle_binary_div(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 1
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 1
        if right == 0:
            raise ZeroDivisionError("division by zero")
        if isinstance(left, float) or isinstance(right, float):
            self.stack.append(left / right)
        else:
            self.stack.append(left // right)

    def _handle_binary_mod(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 1
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 1
        if right == 0:
            raise ZeroDivisionError("division by zero")
        self.stack.append(left % right)

    def _handle_binary_eq(self, operand: Operand) -> None:
        right = self.stack.pop()
        left = self.stack.pop()
        self.stack.append(left == right)

    def _handle_binary_ne(self, operand: Operand) -> None:
        right = self.stack.pop()
        left = self.stack.pop()
        self.stack.append(left != right)

    def _handle_pop(self, operand: Operand) -> None:
        if self.stack:
            self._last_popped = self.stack.pop()

    def _handle_pop_n(self, count: Operand) -> None:
        n = count if count is not None else 1
        for _ in range(n):
            if self.stack:
                self._last_popped = self.stack.pop()

    def _handle_binary_lt(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 0
        self.stack.append(left < right)

    def _handle_binary_le(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 0
        self.stack.append(left <= right)

    def _handle_binary_gt(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 0
        self.stack.append(left > right)

    def _handle_binary_ge(self, operand: Operand) -> None:
        right = self.stack.pop() if self.stack else 0
        left = self.stack.pop() if self.stack else 0
        if left is None:
            left = 0
        if right is None:
            right = 0
        self.stack.append(left >= right)

    def _handle_dup(self, operand: Operand) -> None:
        if self.stack:
            self.stack.append(self.stack[-1])

    # =========================================================================
    # Control Flow & Jumps
    # =========================================================================

    def _handle_jmp(self, offset: Operand) -> None:
        self.pc = offset

    def _handle_jmp_if_true(self, offset: Operand) -> None:
        if self.stack:
            cond = self.stack.pop()
            if cond:
                self.pc = offset

    def _handle_jmp_if_false(self, offset: Operand) -> None:
        if self.stack:
            cond = self.stack.pop()
            if not cond:
                self.pc = offset
        else:
            self.pc = offset

    def _handle_jump_forward(self, offset: Operand) -> None:
        self.pc += offset

    def _handle_pop_jump_if_false(self, offset: Operand) -> None:
        if self.stack:
            cond = self.stack.pop()
            if not cond:
                self.pc = offset
        else:
            self.pc = offset

    def _handle_pop_jump_if_true(self, offset: Operand) -> None:
        if self.stack:
            cond = self.stack.pop()
            if cond:
                self.pc = offset

    def _handle_halt(self, operand: Operand) -> None:
        self.halted = True

    # =========================================================================
    # Return Operations & Non-Local Returns
    # =========================================================================

    def _handle_return_value(self, operand: Operand) -> None:
        """Return value on top of stack. Unwinds blocks up to home_frame during non-local return."""
        return_val = self.stack.pop() if self.stack else None

        if not self.call_stack:
            self._stack.append(return_val)
            self.halted = True
            return

        current = self.call_stack[-1]
        if current.is_block:
            # Non-local return! Unwind to home_frame
            home_frame = current.home_frame
            if (
                home_frame is None
                or not home_frame.is_active
                or home_frame not in self.call_stack
            ):
                raise RuntimeError(
                    "CannotReturn: Block home context has already returned"
                )
            raise NonLocalReturn(return_val, home_frame)
        else:
            if len(self.call_stack) == 1 and current.method_name == "<module>":
                # Top-level root frame: leave return value on stack and halt!
                self.stack.append(return_val)
                self.halted = True
            else:
                self._pop_frame(return_val)

    def _handle_return(self, operand: Operand) -> None:
        """Return from current function/method."""
        if not self.call_stack:
            self.halted = True
            return

        current = self.call_stack[-1]
        if current.is_block:
            self._handle_return_value(operand)
            return

        if len(self.call_stack) == 1:
            self.halted = True
            return

        return_val = self.stack.pop() if self.stack else current.receiver
        self._pop_frame(return_val)

    def _handle_return_block(self, operand: Operand) -> None:
        """Return from block closure to immediate caller."""
        return_val = self.stack.pop() if self.stack else None
        if self.call_stack and self.call_stack[-1].is_block:
            popped = self.call_stack.pop()
            popped.is_active = False
            if self.call_stack:
                self.call_stack[-1].stack.append(return_val)
            else:
                self._stack.append(return_val)
                self.halted = True
        else:
            self._pop_frame(return_val)

    # =========================================================================
    # Collection Indexing & Global Variables
    # =========================================================================

    def _handle_get_index(self, operand: Operand) -> None:
        """Collection indexing read: pop index and collection, push item."""
        if len(self.stack) >= 2:
            index = self.stack.pop()
            collection = self.stack.pop()
            if collection is not None:
                if isinstance(collection, (Dictionary, ActorDictionary)):
                    self.stack.append(collection.at(index))
                elif isinstance(collection, (list, tuple)):
                    idx = int(index)
                    if 0 <= idx < len(collection):
                        self.stack.append(collection[idx])
                    else:
                        self.stack.append(None)
                elif isinstance(collection, dict):
                    self.stack.append(collection.get(index, None))
                elif hasattr(collection, "at") and callable(getattr(collection, "at")):
                    self.stack.append(collection.at(index))
                elif hasattr(collection, "__getitem__"):
                    try:
                        self.stack.append(collection[index])
                    except (IndexError, KeyError):
                        self.stack.append(None)
                else:
                    self.stack.append(None)
            else:
                self.stack.append(None)
        else:
            self.stack.append(None)

    def _handle_set_index(self, operand: Operand) -> None:
        """Collection indexing write: pop value, index, collection, set item."""
        if len(self.stack) >= 3:
            value = self.stack.pop()
            self._last_popped = value
            index = self.stack.pop()
            collection = self.stack.pop()
            if collection is not None:
                if (
                    collection is self.global_vars.get("Actor")
                    or collection is self.global_vars
                ):
                    raw = str(index)
                    norm = raw[1:] if raw.startswith("#") else raw
                    if (
                        self.protect_system_classes
                        and getattr(self, "lock_system_singletons", False)
                    ):
                        from actorii.guardrails.linter import (
                            PROTECTED_POOL_SINGLETONS,
                            ProtectedClassError,
                        )

                        if norm in PROTECTED_POOL_SINGLETONS and (
                            norm in collection or raw in collection
                        ):
                            raise ProtectedClassError(
                                f"ProtectedSingletonError: Cannot overwrite locked system singleton 'Actor[#{norm}]'"
                            )
                if isinstance(collection, (Dictionary, ActorDictionary)):
                    collection.atPut(index, value)
                elif isinstance(collection, list):
                    idx = int(index)
                    if 0 <= idx < len(collection):
                        collection[idx] = value
                    elif idx >= len(collection):
                        collection.extend([None] * (idx - len(collection) + 1))
                        collection[idx] = value
                elif isinstance(collection, dict):
                    if (
                        collection is self.global_vars.get("Actor")
                        or collection is self.global_vars
                    ):
                        raw = str(index)
                        norm = raw[1:] if raw.startswith("#") else raw
                        if (
                            self.protect_system_classes
                            and getattr(self, "lock_system_singletons", False)
                        ):
                            from actorii.guardrails.linter import (
                                PROTECTED_POOL_SINGLETONS,
                                ProtectedClassError,
                            )

                            if norm in PROTECTED_POOL_SINGLETONS and (
                                norm in collection or raw in collection
                            ):
                                raise ProtectedClassError(
                                    f"ProtectedSingletonError: Cannot overwrite locked system singleton 'Actor[#{norm}]'"
                                )
                        collection[index] = value
                        collection[norm] = value
                        collection[raw] = value
                        self.global_vars[norm] = value
                        self.global_vars[raw] = value
                    else:
                        collection[index] = value
                elif hasattr(collection, "atPut") and callable(
                    getattr(collection, "atPut")
                ):
                    collection.atPut(index, value)
                elif hasattr(collection, "put") and callable(
                    getattr(collection, "put")
                ):
                    collection.put(value, index)
                elif hasattr(collection, "__setitem__"):
                    try:
                        collection[index] = value
                    except (IndexError, KeyError):
                        pass
            if operand != 0:
                self.stack.append(value)

    # =========================================================================
    # Primitives & Built-in Support
    # =========================================================================

    def _handle_invoke_prim(self, prim_idx: Operand) -> None:
        """Invoke built-in runtime primitive (max, min, sqrt, asInt, dec, inc, fill, abs)."""
        desc = self.constants[prim_idx] if prim_idx < len(self.constants) else {}
        prim_name = ""
        arg_count = 1
        if isinstance(desc, dict):
            prim_name = desc.get("name", "")
            arg_count = desc.get("args", 1)
        else:
            prim_name = str(desc)
            if prim_name in ("max", "min", "fill"):
                arg_count = 2
            elif prim_name in ("sqrt", "asInt", "dec", "inc", "abs"):
                arg_count = 1

        args = (
            [self.stack.pop() for _ in range(arg_count)][::-1]
            if len(self.stack) >= arg_count
            else []
        )
        result = None
        if prim_name == "max":
            result = (
                max(args[0], args[1]) if len(args) >= 2 else (args[0] if args else None)
            )
        elif prim_name == "min":
            result = (
                min(args[0], args[1]) if len(args) >= 2 else (args[0] if args else None)
            )
        elif prim_name == "sqrt":
            result = math.sqrt(args[0]) if args else 0.0
        elif prim_name == "asInt":
            result = int(args[0]) if args else 0
        elif prim_name == "dec":
            result = (args[0] - 1) if args else 0
        elif prim_name == "inc":
            result = (args[0] + 1) if args else 0
        elif prim_name == "abs":
            result = abs(args[0]) if args else 0
        elif prim_name == "fill":
            if len(args) >= 2:
                coll = args[0]
                val = args[1]
                if isinstance(coll, list):
                    for i in range(len(coll)):
                        coll[i] = val
                    result = coll
                elif hasattr(coll, "fill"):
                    result = coll.fill(val)
                else:
                    result = coll
            else:
                result = args[0] if args else None

        self.stack.append(result)

    # =========================================================================
    # Block Closures
    # =========================================================================

    def _handle_make_block(self, block_idx: Operand) -> None:
        """Instantiate block closure object and push onto stack."""
        block_desc = (
            self.constants[block_idx] if block_idx < len(self.constants) else {}
        )
        outer_frame = self.current_frame
        home_frame = (
            outer_frame.home_frame
            if (outer_frame and outer_frame.is_block)
            else outer_frame
        )
        receiver = outer_frame.receiver if outer_frame else None

        captured_cells: Dict[int, Cell] = {}
        captured_vars = getattr(block_desc, "captured_vars", [])
        if outer_frame and captured_vars:
            for item in captured_vars:
                if isinstance(item, tuple) and len(item) >= 3:
                    name = item[0]
                    depth = item[1]
                    outer_slot = item[2]
                    inner_slot = getattr(item, "inner_slot", None)
                    if inner_slot is None:
                        inner_slot = item[3] if len(item) > 3 else outer_slot
                else:
                    continue

                target = outer_frame
                while len(target.local_vars) <= outer_slot:
                    target.local_vars.append(None)

                val = target.local_vars[outer_slot]
                if not isinstance(val, Cell):
                    cell = Cell(val)
                    target.local_vars[outer_slot] = cell
                else:
                    cell = val

                captured_cells[inner_slot] = cell

        closure = BlockClosure(
            descriptor=block_desc,
            outer_frame=outer_frame,
            home_frame=home_frame,
            receiver=receiver,
            captured_cells=captured_cells,
        )
        self.stack.append(closure)

    def _handle_call_block(self, argc: Operand) -> None:
        """Invoke block closure."""
        args = (
            [self.stack.pop() for _ in range(argc)][::-1]
            if len(self.stack) >= argc
            else []
        )
        closure = self.stack.pop() if self.stack else None

        if closure is None or not isinstance(closure, BlockClosure):
            self.stack.append(None)
            return

        desc = closure.descriptor
        desc_locals = getattr(desc, "locals", [])

        block_locals: List[Any] = [closure.receiver]
        block_locals.extend(args)
        block_locals.extend([None] * len(desc_locals))

        for inner_slot, cell in closure.captured_cells.items():
            while len(block_locals) <= inner_slot:
                block_locals.append(None)
            block_locals[inner_slot] = cell

        block_frame = StackFrame(
            instructions=closure.instructions,
            pc=0,
            constants=closure.constants or self.constants,
            strings=closure.string_table or self.string_table,
            local_vars=block_locals,
            receiver=closure.receiver,
            caller_frame=self.current_frame,
            method_name=getattr(desc, "name", "Block"),
            class_name=closure.home_frame.class_name if closure.home_frame else "",
            is_block=True,
            stack=[],
            home_frame=closure.home_frame,
            outer_frame=closure.outer_frame,
            closure=closure,
        )

        self._push_frame(block_frame)

        if not self._in_run_loop:
            target_depth = len(self.call_stack) - 1
            while len(self.call_stack) > target_depth and not self.halted:
                f = self.call_stack[-1]
                if f.is_exhausted:
                    self._pop_frame(None)
                    continue
                op, arg = f.instructions[f.pc]
                f.pc += 1
                self._execute_opcode(op, arg)

    # =========================================================================
    # Method Dispatch & Invocation
    # =========================================================================

    def _handle_call_early(self, early_idx: Operand) -> None:
        """Execute early-bound method call bypassing dynamic subclass override."""
        desc = self.constants[early_idx] if early_idx < len(self.constants) else {}
        if isinstance(desc, dict):
            class_name = desc.get("target_class") or desc.get("class") or ""
            method_name = desc.get("method", "")
            arg_count = desc.get("args", desc.get("arg_count", 0))
        elif hasattr(desc, "target_class") or hasattr(desc, "method"):
            class_name = getattr(desc, "target_class", getattr(desc, "class_name", ""))
            method_name = getattr(desc, "method", "")
            arg_count = getattr(desc, "args", getattr(desc, "arg_count", 0))
        else:
            class_name = str(desc)
            method_name = ""
            arg_count = 0

        args = (
            [self.stack.pop() for _ in range(arg_count)][::-1]
            if len(self.stack) >= arg_count
            else []
        )
        receiver = self.stack.pop() if self.stack else None

        if method_name == "at" and isinstance(receiver, (list, tuple)):
            idx = int(args[0])
            self.stack.append(receiver[idx] if 0 <= idx < len(receiver) else None)
            return
        elif method_name == "put" and isinstance(receiver, list):
            val = args[0]
            idx = int(args[1]) if len(args) > 1 else 0
            if 0 <= idx < len(receiver):
                receiver[idx] = val
            elif idx >= len(receiver):
                receiver.extend([None] * (idx - len(receiver) + 1))
                receiver[idx] = val
            self.stack.append(val)
            return

        cls = self.hierarchy.get(class_name) if hasattr(self.hierarchy, "get") else None
        method = None
        if hasattr(self.hierarchy, "resolve_method"):
            method = self.hierarchy.resolve_method(class_name, method_name)
        elif cls and hasattr(cls, "get_method") and cls.has_method(method_name):
            method = cls.get_method(method_name)

        if method is not None and callable(method):
            try:
                result = method(receiver, *args)
            except TypeError:
                try:
                    result = method(*args)
                except Exception:
                    result = None
            self.stack.append(result)
        elif receiver and hasattr(receiver, method_name):
            attr = getattr(receiver, method_name)
            res = attr(*args) if callable(attr) else attr
            self.stack.append(res)
        else:
            self.stack.append(None)

    def _handle_call_global(self, func_idx: Operand) -> None:
        """Call a global function or constructor."""
        if func_idx < len(self.constants):
            name = str(self.constants[func_idx])
            func = self.global_vars.get(name)
            if func is not None and callable(func):
                res = func()
                self.stack.append(res)
            elif hasattr(self.hierarchy, "get"):
                cls = self.hierarchy.get(name)
                if cls is not None:
                    inst = cls._default_constructor()
                    self.stack.append(inst)

    def _handle_call_method(self, method_idx: Operand) -> None:
        """Execute dynamic message send: receiver-based polymorphic method dispatch."""
        desc = self.constants[method_idx] if method_idx < len(self.constants) else None
        method_name = ""
        arg_count: Optional[int] = None

        if hasattr(desc, "arg_count") and getattr(desc, "arg_count") is not None:
            arg_count = getattr(desc, "arg_count")
            method_name = getattr(desc, "method", getattr(desc, "name", str(desc)))
        elif isinstance(desc, dict):
            method_name = desc.get("method") or desc.get("name", "")
            arg_count = desc.get("args", desc.get("arg_count", None))
        elif hasattr(desc, "method"):
            method_name = getattr(desc, "method")
            arg_count = getattr(desc, "args", getattr(desc, "arg_count", None))
        elif hasattr(desc, "name"):
            method_name = getattr(desc, "name")
            arg_count = getattr(desc, "args", getattr(desc, "arg_count", None))
        elif isinstance(desc, str):
            method_name = desc
        else:
            method_name = str(desc) if desc is not None else ""

        if arg_count is not None:
            args = (
                [self.stack.pop() for _ in range(arg_count)][::-1]
                if len(self.stack) >= arg_count
                else []
            )
            receiver = self.stack.pop() if self.stack else None
        else:
            if len(self.stack) <= 1:
                receiver = self.stack.pop() if self.stack else None
                args = []
            else:
                found = False
                for candidate_argc in range(len(self.stack) - 1, -1, -1):
                    cand_receiver = self.stack[-1 - candidate_argc]
                    expected_argc = self._infer_method_arity(cand_receiver, method_name)
                    if expected_argc == candidate_argc:
                        args = [self.stack.pop() for _ in range(candidate_argc)][::-1]
                        receiver = self.stack.pop()
                        found = True
                        break
                if not found:
                    candidate_argc = self._infer_method_arity(None, method_name)
                    if (
                        candidate_argc is not None
                        and len(self.stack) >= candidate_argc + 1
                    ):
                        args = [self.stack.pop() for _ in range(candidate_argc)][::-1]
                        receiver = self.stack.pop() if self.stack else None
                    elif len(self.stack) >= 2:
                        args = [self.stack.pop()]
                        receiver = self.stack.pop()
                    else:
                        receiver = self.stack.pop() if self.stack else None
                        args = []

        # Built-in dispatch for primitive types
        if isinstance(receiver, int):
            if method_name == "over":
                stop = args[0] if args else 0
                self.stack.append(Interval(receiver, stop, 1))
                return
            elif method_name == "overBy":
                stop = args[0] if args else 0
                step = args[1] if len(args) > 1 else 1
                self.stack.append(Interval(receiver, stop, step))
                return
            elif method_name == "do":
                interval = Interval(0, receiver, 1)
                self._run_do_loop(interval, args[0] if args else None)
                self.stack.append(None)
                return
            elif method_name == "point3D":
                yv = args[0] if args else 0
                zv = args[1] if len(args) > 1 else 0
                self._dispatch_int_point3d(receiver, yv, zv)
                return
            elif method_name == "sqrt":
                self.stack.append(math.sqrt(receiver))
                return
            elif method_name == "asInt":
                self.stack.append(int(receiver))
                return
            elif method_name == "dec":
                self.stack.append(receiver - 1)
                return
            elif method_name == "inc":
                self.stack.append(receiver + 1)
                return
            elif method_name == "abs":
                self.stack.append(abs(receiver))
                return
        elif isinstance(receiver, Interval):
            if method_name == "do":
                last = self._run_do_loop(receiver, args[0] if args else None)
                self.stack.append(last if last is not None else receiver)
                return
            elif method_name == "collect":
                res = receiver.collect(args[0] if args else None)
                self.stack.append(res)
                return
            elif method_name == "at":
                self.stack.append(receiver.at(args[0]))
                return
            elif method_name == "size":
                self.stack.append(receiver.size())
                return
            elif method_name == "first":
                self.stack.append(receiver.first())
                return
            elif method_name == "last":
                self.stack.append(receiver.last())
                return
            elif method_name == "start":
                self.stack.append(receiver.start)
                return
            elif method_name == "stop":
                self.stack.append(receiver.stop)
                return
            elif method_name == "step":
                self.stack.append(receiver.step)
                return
            elif method_name == "in":
                self.stack.append(receiver.in_interval(args[0]))
                return
            elif method_name == "asArray":
                self.stack.append(receiver.asArray())
                return
            elif method_name == "reverse":
                self.stack.append(receiver.reverse())
                return
        elif isinstance(receiver, (Dictionary, ActorDictionary, dict)):
            is_actor_pool = (
                receiver is self.global_vars.get("Actor")
                or receiver is self.global_vars
            )
            if is_actor_pool and self.protect_system_classes and getattr(self, "lock_system_singletons", False):
                from actorii.guardrails.linter import PROTECTED_POOL_SINGLETONS, ProtectedClassError

                if method_name == "clear":
                    raise ProtectedClassError("ProtectedSingletonError: Cannot clear locked Actor[...] singleton pool")
                if method_name in ("atPut", "put", "add", "removeKey", "remove", "removeAt") and args:
                    key_arg = args[1] if (method_name == "put" and len(args) > 1 and str(args[1]).lstrip("#") in PROTECTED_POOL_SINGLETONS) else args[0]
                    raw_k = str(key_arg)
                    norm_k = raw_k[1:] if raw_k.startswith("#") else raw_k
                    if norm_k in PROTECTED_POOL_SINGLETONS:
                        raise ProtectedClassError(
                            f"ProtectedSingletonError: Cannot mutate locked system singleton 'Actor[#{norm_k}]' via {method_name}"
                        )
            if isinstance(receiver, dict):
                if method_name == "at":
                    k = args[0]
                    raw_k = str(k)
                    norm_k = raw_k[1:] if raw_k.startswith("#") else raw_k
                    self.stack.append(receiver.get(k, receiver.get(norm_k, receiver.get(raw_k))))
                    return
                elif method_name in ("atPut", "put", "add"):
                    k, v = args[0], args[1]
                    receiver[k] = v
                    if is_actor_pool:
                        raw_k = str(k)
                        norm_k = raw_k[1:] if raw_k.startswith("#") else raw_k
                        receiver[norm_k] = v
                        receiver[raw_k] = v
                    self.stack.append(v)
                    return
                elif method_name in ("removeKey", "remove", "removeAt"):
                    k = args[0]
                    self.stack.append(receiver.pop(k, None))
                    return
                elif method_name == "size":
                    self.stack.append(len(receiver))
                    return
                elif method_name == "isEmpty":
                    self.stack.append(len(receiver) == 0)
                    return
                elif method_name == "keys":
                    self.stack.append(list(receiver.keys()))
                    return
                elif method_name == "values":
                    self.stack.append(list(receiver.values()))
                    return
            if method_name == "at":
                self.stack.append(receiver.at(args[0]))
                return
            elif method_name == "atPut":
                self.stack.append(receiver.atPut(args[0], args[1]))
                return
            elif method_name == "put":
                self.stack.append(receiver.put(args[0], args[1]))
                return
            elif method_name == "add":
                self.stack.append(receiver.add(args[0], args[1]))
                return
            elif method_name == "includesKey":
                self.stack.append(receiver.includesKey(args[0]))
                return
            elif method_name == "keys":
                self.stack.append(receiver.keys())
                return
            elif method_name == "values":
                self.stack.append(receiver.values())
                return
            elif method_name in ("removeKey", "remove", "removeAt"):
                self.stack.append(receiver.removeKey(args[0]))
                return
            elif method_name == "size":
                self.stack.append(receiver.size())
                return
            elif method_name == "isEmpty":
                self.stack.append(receiver.isEmpty())
                return
            elif method_name == "clear":
                receiver.clear()
                self.stack.append(receiver)
                return
            elif method_name == "do":
                block = args[0] if args else None
                last = None
                for val in receiver.values():
                    last = (
                        self._call_block_closure(block, [val])
                        if isinstance(block, BlockClosure)
                        else (block(val) if callable(block) else val)
                    )
                self.stack.append(last if last is not None else receiver)
                return
            elif method_name == "keysDo":
                block = args[0] if args else None
                last = None
                for k in receiver.keys():
                    last = (
                        self._call_block_closure(block, [k])
                        if isinstance(block, BlockClosure)
                        else (block(k) if callable(block) else k)
                    )
                self.stack.append(last)
                return
            elif method_name == "assocsDo":
                block = args[0] if args else None
                last = None
                items = (
                    list(receiver._entries.items())
                    if hasattr(receiver, "_entries")
                    else list(receiver.items())
                )
                for k, v in items:
                    last = (
                        self._call_block_closure(block, [[k, v]])
                        if isinstance(block, BlockClosure)
                        else (block([k, v]) if callable(block) else [k, v])
                    )
                self.stack.append(last)
                return
        elif isinstance(receiver, (list, tuple, ActorIndexedCollection)):
            if method_name == "size":
                self.stack.append(len(receiver))
                return
            elif method_name == "isEmpty":
                self.stack.append(len(receiver) == 0)
                return
            elif method_name == "at":
                idx = int(args[0])
                self.stack.append(receiver[idx] if 0 <= idx < len(receiver) else None)
                return
            elif method_name == "atPut":
                idx = int(args[0])
                val = args[1]
                if hasattr(receiver, "atPut"):
                    self.stack.append(receiver.atPut(idx, val))
                else:
                    if 0 <= idx < len(receiver):
                        receiver[idx] = val
                    elif idx >= len(receiver):
                        receiver.extend([None] * (idx - len(receiver) + 1))
                        receiver[idx] = val
                    self.stack.append(val)
                return
            elif method_name == "put":
                if isinstance(receiver, list):
                    val = args[0]
                    idx = int(args[1]) if len(args) > 1 else 0
                    if hasattr(receiver, "put"):
                        self.stack.append(receiver.put(val, idx))
                    else:
                        if 0 <= idx < len(receiver):
                            receiver[idx] = val
                        elif idx >= len(receiver):
                            receiver.extend([None] * (idx - len(receiver) + 1))
                            receiver[idx] = val
                        self.stack.append(val)
                else:
                    self.stack.append(None)
                return
            elif method_name in ("add", "push", "addLast"):
                item = args[0] if args else None
                if hasattr(receiver, "add"):
                    self.stack.append(receiver.add(item))
                elif isinstance(receiver, list):
                    receiver.append(item)
                    self.stack.append(item)
                return
            elif method_name == "addFirst":
                item = args[0] if args else None
                if hasattr(receiver, "addFirst"):
                    self.stack.append(receiver.addFirst(item))
                elif isinstance(receiver, list):
                    receiver.insert(0, item)
                    self.stack.append(item)
                return
            elif method_name == "first":
                if hasattr(receiver, "first"):
                    self.stack.append(receiver.first())
                elif len(receiver) > 0:
                    self.stack.append(receiver[0])
                else:
                    raise IndexError("Empty collection")
                return
            elif method_name == "last":
                if hasattr(receiver, "last"):
                    self.stack.append(receiver.last())
                elif len(receiver) > 0:
                    self.stack.append(receiver[-1])
                else:
                    raise IndexError("Empty collection")
                return
            elif method_name in ("removeLast", "pop"):
                if hasattr(receiver, "pop"):
                    self.stack.append(receiver.pop(*args))
                elif isinstance(receiver, list):
                    self.stack.append(receiver.pop(*args))
                return
            elif method_name == "removeFirst":
                if hasattr(receiver, "removeFirst"):
                    self.stack.append(receiver.removeFirst())
                elif isinstance(receiver, list):
                    self.stack.append(receiver.pop(0))
                return
            elif method_name == "remove":
                if hasattr(receiver, "remove"):
                    self.stack.append(receiver.remove(args[0]))
                elif isinstance(receiver, list):
                    if isinstance(args[0], int) and 0 <= args[0] < len(receiver):
                        self.stack.append(receiver.pop(args[0]))
                    else:
                        receiver.remove(args[0])
                        self.stack.append(args[0])
                return
            elif method_name == "removeAt":
                if hasattr(receiver, "removeAt"):
                    self.stack.append(receiver.removeAt(args[0]))
                elif isinstance(receiver, list):
                    self.stack.append(receiver.pop(int(args[0])))
                return
            elif method_name == "insert":
                if hasattr(receiver, "insert"):
                    self.stack.append(receiver.insert(*args))
                elif isinstance(receiver, list):
                    receiver.insert(int(args[0]), args[1])
                    self.stack.append(args[1])
                return
            elif method_name == "fill":
                val = args[0] if args else None
                if hasattr(receiver, "fill"):
                    self.stack.append(receiver.fill(val))
                elif isinstance(receiver, list):
                    for i in range(len(receiver)):
                        receiver[i] = val
                    self.stack.append(receiver)
                return
            elif method_name == "copyFrom":
                b = int(args[0])
                e = int(args[1])
                if hasattr(receiver, "copyFrom"):
                    self.stack.append(receiver.copyFrom(b, e))
                elif isinstance(receiver, (list, tuple)):
                    self.stack.append(list(receiver[b:e]))
                return
            elif method_name == "do":
                last = self._run_do_loop(receiver, args[0] if args else None)
                self.stack.append(last if last is not None else receiver)
                return
            elif method_name == "collect":
                block = args[0] if args else None
                res = [
                    (
                        self._call_block_closure(block, [x])
                        if isinstance(block, BlockClosure)
                        else (block(x) if callable(block) else x)
                    )
                    for x in receiver
                ]
                self.stack.append(ActorOrderedCollection(res))
                return
            elif method_name in ("select", "extract"):
                block = args[0] if args else None
                res = []
                for x in receiver:
                    val = (
                        self._call_block_closure(block, [x])
                        if isinstance(block, BlockClosure)
                        else (block(x) if callable(block) else x)
                    )
                    if val is not None and val is not False and val != 0:
                        res.append(x)
                self.stack.append(ActorOrderedCollection(res))
                return
            elif method_name == "reject":
                block = args[0] if args else None
                res = []
                for x in receiver:
                    val = (
                        self._call_block_closure(block, [x])
                        if isinstance(block, BlockClosure)
                        else (block(x) if callable(block) else x)
                    )
                    if val is None or val is False or val == 0:
                        res.append(x)
                self.stack.append(ActorOrderedCollection(res))
                return
            elif method_name == "asArray":
                if hasattr(receiver, "asArray"):
                    self.stack.append(receiver.asArray())
                else:
                    self.stack.append(ActorArray(receiver))
                return
            elif method_name == "asOrderedCollection":
                if hasattr(receiver, "asOrderedCollection"):
                    self.stack.append(receiver.asOrderedCollection())
                else:
                    self.stack.append(ActorOrderedCollection(receiver))
                return
            elif method_name == "keysDo":
                block = args[0] if args else None
                last = None
                for i in range(len(receiver)):
                    last = (
                        self._call_block_closure(block, [i])
                        if isinstance(block, BlockClosure)
                        else (block(i) if callable(block) else i)
                    )
                self.stack.append(last)
                return
            elif method_name == "indexOf":
                target = args[0] if args else None
                idx = None
                for i, elem in enumerate(receiver):
                    if elem is target:
                        idx = i
                        break
                self.stack.append(idx)
                return
            elif method_name == "find":
                target = args[0] if args else None
                idx = None
                for i, elem in enumerate(receiver):
                    if elem == target:
                        idx = i
                        break
                self.stack.append(idx)
                return
            elif method_name == "reverse":
                if hasattr(receiver, "reverse"):
                    receiver.reverse()
                    self.stack.append(receiver)
                elif isinstance(receiver, list):
                    receiver.reverse()
                    self.stack.append(receiver)
                return
        elif isinstance(receiver, (Point, Point3D)):
            if method_name == "x":
                self.stack.append(receiver.x)
                return
            elif method_name == "y":
                self.stack.append(receiver.y)
                return
            elif method_name == "z":
                self.stack.append(getattr(receiver, "z", 0))
                return
            elif method_name == "setX":
                self.stack.append(receiver.setX(args[0]))
                return
            elif method_name == "setY":
                self.stack.append(receiver.setY(args[0]))
                return
            elif method_name == "setZ" and hasattr(receiver, "setZ"):
                self.stack.append(receiver.setZ(args[0]))
                return
            elif method_name == "set":
                self.stack.append(receiver.set(args[0], args[1]))
                return
            elif method_name == "asPoint":
                self.stack.append(receiver.asPoint())
                return
            elif method_name == "middlePoint" and hasattr(receiver, "middlePoint"):
                rng = args[1] if len(args) > 1 else 0
                self.stack.append(receiver.middlePoint(args[0], rng))
                return
            elif method_name == "round":
                self.stack.append(receiver.round())
                return
        elif isinstance(receiver, ActorClass):
            if method_name == "new":
                if receiver.name == "Array":
                    sz = int(args[0]) if args else 0
                    self.stack.append(ActorArray(sz))
                    return
                elif receiver.name == "OrderedCollection":
                    self.stack.append(ActorOrderedCollection(args[0] if args else None))
                    return
                elif receiver.name == "List":
                    self.stack.append(ActorList(args[0] if args else None))
                    return
                elif receiver.name == "Dictionary":
                    cap = int(args[0]) if args else 16
                    self.stack.append(Dictionary(cap))
                    return
                elif receiver.name == "Interval":
                    s = int(args[0]) if len(args) > 0 else 0
                    e = int(args[1]) if len(args) > 1 else 0
                    st = int(args[2]) if len(args) > 2 else 1
                    self.stack.append(Interval(s, e, st))
                    return
                elif receiver.name == "Point":
                    x = int(args[0]) if len(args) > 0 else 0
                    y = int(args[1]) if len(args) > 1 else 0
                    self.stack.append(Point(x, y))
                    return
                elif receiver.name == "Point3D":
                    x = int(args[0]) if len(args) > 0 else 0
                    y = int(args[1]) if len(args) > 1 else 0
                    z = int(args[2]) if len(args) > 2 else 0
                    self.stack.append(Point3D(x, y, z))
                    return
                elif receiver.name == "Rect":
                    self.stack.append(Rect(*args))
                    return
                elif receiver.name == "ThermometerUI" or self._is_subclass_of(receiver.name, "ThermometerUI"):
                    inst = ThermometerUI(*args, class_name=receiver.name, vm=self)
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "Dialog" or self._is_subclass_of(receiver.name, "Dialog"):
                    inst = Dialog(*args)
                    inst._class_name = receiver.name
                    inst.vm = self
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "Window" or self._is_subclass_of(receiver.name, "Window"):
                    inst = Window(*args)
                    inst._class_name = receiver.name
                    inst.vm = self
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "Checkbox" or self._is_subclass_of(receiver.name, "Checkbox"):
                    inst = Checkbox(*args)
                    inst._class_name = receiver.name
                    inst.vm = self
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "Button" or self._is_subclass_of(receiver.name, "Button"):
                    inst = Button(*args)
                    inst._class_name = receiver.name
                    inst.vm = self
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "TextField" or self._is_subclass_of(receiver.name, "TextField"):
                    inst = TextField(*args)
                    inst._class_name = receiver.name
                    inst.vm = self
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "Control" or self._is_subclass_of(receiver.name, "Control"):
                    inst = Control(*args)
                    inst._class_name = receiver.name
                    inst.vm = self
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "Set":
                    self.stack.append(ActorSet(args[0] if args else None))
                    return
                elif receiver.name == "Agent" or self._is_subclass_of(receiver.name, "Agent"):
                    inst = Agent(*args, class_name=receiver.name, vm=self)
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                elif receiver.name == "ThermoProTwin" or self._is_subclass_of(receiver.name, "ThermoProTwin"):
                    inst = ThermoProTwin(*args, class_name=receiver.name, vm=self)
                    inst.inherit_from(receiver)
                    for iv in self._collect_class_ivars(receiver.name):
                        inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
                else:
                    if (
                        hasattr(receiver, "_default_constructor")
                        and receiver._default_constructor is not None
                    ):
                        inst = receiver._default_constructor(*args)
                    else:
                        inst = ActorObject(obj_id=str(id(receiver)))
                        inst._class_name = receiver.name
                        inst.inherit_from(receiver)
                    inst._class_name = receiver.name
                    if hasattr(inst, "_fields") and isinstance(inst._fields, dict):
                        for iv in self._collect_class_ivars(receiver.name):
                            inst._fields.setdefault(iv, None)
                    self.stack.append(inst)
                    return
        elif isinstance(receiver, type):
            if method_name == "new":
                self.stack.append(receiver(*args))
                return
            elif hasattr(receiver, method_name):
                attr = getattr(receiver, method_name)
                if callable(attr):
                    self.stack.append(attr(*args))
                    return
        elif isinstance(receiver, BlockClosure):
            if method_name in ("eval", "value", "call"):
                res = self._call_block_closure(receiver, args)
                self.stack.append(res)
                return

        # Built-in 1993 Image Persistence (saveImage, loadImage, restoreImage, snapshot)
        if method_name in ("saveImage", "snapshot"):
            img_path = (
                str(args[0])
                if args
                else (str(receiver) if isinstance(receiver, str) else "ACTOR.IMG")
            )
            saved_path = self.save_image(img_path)
            self.stack.append(saved_path)
            return
        elif method_name in ("loadImage", "restoreImage"):
            img_path = (
                str(args[0])
                if args
                else (str(receiver) if isinstance(receiver, str) else "ACTOR.IMG")
            )
            stats = self.load_image(img_path)
            self.stack.append(stats.get("path", img_path))
            return
        elif method_name == "enableJitSynthesis":
            flag = bool(args[0]) if args else True
            if isinstance(receiver, ActorClass):
                receiver.jit_synthesis = flag
                self.stack.append(receiver)
            elif isinstance(receiver, ActorObject):
                receiver._fields["autoSynthesize"] = flag
                receiver.auto_synthesize = flag
                self.stack.append(receiver)
            else:
                self.jit_synthesis_enabled = bool(receiver) if not args else flag
                self.stack.append(self.jit_synthesis_enabled)
            return
        elif method_name == "getField":
            fname = str(args[0]) if args else ""
            if fname.startswith("#"):
                fname = fname[1:]
            val = None
            if hasattr(receiver, "_fields") and isinstance(receiver._fields, dict):
                val = receiver._fields.get(fname)
            elif hasattr(receiver, fname):
                val = getattr(receiver, fname)
            self.stack.append(val)
            return
        elif method_name == "setField":
            fname = str(args[0]) if args else ""
            if fname.startswith("#"):
                fname = fname[1:]
            fval = args[1] if len(args) > 1 else None
            if hasattr(receiver, "_fields") and isinstance(receiver._fields, dict):
                receiver._fields[fname] = fval
            elif hasattr(receiver, fname):
                try:
                    setattr(receiver, fname, fval)
                except Exception:
                    pass
            self.stack.append(fval)
            return
        elif method_name == "synthesizedSource":
            sel = str(args[0]) if args else ""
            if sel.startswith("#"):
                sel = sel[1:]
            target_cls_name = (
                receiver.name
                if isinstance(receiver, ActorClass)
                else self._get_class_name(receiver)
            )
            m_obj = self.hierarchy.resolve_method(target_cls_name, sel)
            if m_obj is None and hasattr(receiver, "get_method"):
                m_obj = receiver.get_method(sel)
            self.stack.append(getattr(m_obj, "source", None))
            return

        # Polymorphic method lookup
        class_name = self._get_class_name(receiver)
        method = None
        if class_name and hasattr(self.hierarchy, "resolve_method"):
            method = self.hierarchy.resolve_method(class_name, method_name)

        if (
            method is None
            and hasattr(receiver, "get_method")
            and not isinstance(receiver, type)
        ):
            method = receiver.get_method(method_name)

        is_bridge_receiver = hasattr(receiver, "_bridge_specs")
        if (
            method is None
            and not method_name.startswith("_")
            and hasattr(receiver, method_name)
        ):
            if not is_bridge_receiver or method_name in getattr(
                receiver, "_STANDARD_OBJECT_SELECTORS", ()
            ):
                attr = getattr(receiver, method_name)
                if callable(attr):
                    method = attr

        # 1993 Actor `doesNotUnderstand` / `fail` hook on the class hierarchy
        if method is None and class_name and hasattr(self.hierarchy, "resolve_method"):
            dnu_method = self.hierarchy.resolve_method(
                class_name, "doesNotUnderstand"
            ) or self.hierarchy.resolve_method(class_name, "fail")
            if (
                dnu_method is None
                and hasattr(receiver, "get_method")
                and not isinstance(receiver, type)
            ):
                dnu_method = receiver.get_method(
                    "doesNotUnderstand"
                ) or receiver.get_method("fail")
            if dnu_method is not None:
                method = dnu_method
                dnu_params = getattr(dnu_method, "params", [])
                if len(dnu_params) == 1:
                    args = [Symbol(method_name)]
                elif len(dnu_params) >= 3:
                    args = [None, Symbol(method_name), ActorArray(args)]
                else:
                    args = [Symbol(method_name), ActorArray(args)]
                method_name = getattr(dnu_method, "name", "doesNotUnderstand")

        # AI + OOP Dynamic JIT Method Synthesis on `doesNotUnderstand`
        if method is None and self._should_jit_synthesize(
            receiver, class_name, method_name
        ):
            method = self._jit_synthesize_method(
                receiver, class_name, method_name, args
            )

        if method is not None and (
            isinstance(method, CompiledMethod) or hasattr(method, "instructions")
        ):
            params = getattr(method, "params", [])
            declared_locals = getattr(method, "locals", [])

            local_vars: List[Any] = [receiver]
            local_vars.extend(args)
            if len(args) < len(params):
                local_vars.extend([None] * (len(params) - len(args)))
            local_vars.extend([None] * len(declared_locals))

            callee_frame = StackFrame(
                instructions=method.instructions,
                pc=0,
                constants=getattr(method, "constants", self.constants),
                strings=getattr(
                    method,
                    "string_table",
                    getattr(method, "strings", self.string_table),
                ),
                local_vars=local_vars,
                receiver=receiver,
                caller_frame=self.current_frame,
                method_name=method_name,
                class_name=class_name,
                is_block=False,
                stack=[],
            )
            self._push_frame(callee_frame)

            if not self._in_run_loop:
                target_depth = len(self.call_stack) - 1
                try:
                    while len(self.call_stack) > target_depth and not self.halted:
                        f = self.call_stack[-1]
                        if f.is_exhausted:
                            self._pop_frame(f.receiver)
                            continue
                        op, arg = f.instructions[f.pc]
                        f.pc += 1
                        self._execute_opcode(op, arg)
                except NonLocalReturn as nlr:
                    while self.call_stack and self.call_stack[-1] is not nlr.home_frame:
                        popped = self.call_stack.pop()
                        popped.is_active = False
                    if self.call_stack and self.call_stack[-1] is nlr.home_frame:
                        popped = self.call_stack.pop()
                        popped.is_active = False
                    if self.call_stack:
                        self.call_stack[-1].stack.append(nlr.value)
                    else:
                        self._stack.append(nlr.value)
                    if self.call_stack and nlr.home_frame in self.call_stack:
                        raise
                except Exception:
                    if self.call_stack and getattr(self, "_last_failed_frame_info", None) is None:
                        top_f = self.call_stack[-1]
                        cls_obj = self.hierarchy.get(top_f.class_name) if top_f.class_name else None
                        m_obj = cls_obj.get_method(top_f.method_name) if cls_obj else None
                        var_names = ["self"] + list(getattr(m_obj, "params", []) or []) + list(getattr(m_obj, "locals", []) or [])
                        local_map = {}
                        for idx_l, lv in enumerate(top_f.local_vars):
                            if idx_l == 0:
                                continue
                            v_name = var_names[idx_l] if idx_l < len(var_names) else f"local_{idx_l}"
                            raw_v = lv.value if isinstance(lv, Cell) else lv
                            local_map[str(v_name)] = repr(raw_v)[:120]
                        self._last_failed_frame_info = {
                            "failedClass": str(top_f.class_name or "<Chunk>"),
                            "failedSelector": str(top_f.method_name or "<chunk>"),
                            "bytecodeOffset": max(0, int(top_f.pc) - 1),
                            "localBindings": local_map,
                        }
                    while len(self.call_stack) > target_depth:
                        popped = self.call_stack.pop()
                        popped.is_active = False
                    raise

        elif method is not None and callable(method):
            result = self._invoke_python_callable(method, receiver, args)
            self.stack.append(result)
        else:
            if receiver is not None and hasattr(receiver, "message_not_understood"):
                res = receiver.message_not_understood(method_name, *args)
                self.stack.append(res)
            elif method_name in ("atPut", "put") and isinstance(receiver, (int, float, str, bool)):
                raise TypeError(
                    f"`{method_name}` cannot be called on scalar `{receiver!r}` (type={type(receiver).__name__}). "
                    "Hint: `atPut(dict, #key, val)` mutates `dict` in-place and returns `val` (NOT `dict`). "
                    "Do NOT chain `d := atPut(d, ...)` — initialize `d := new(Dictionary, 8)!!` first, then call `atPut(d, #k1, v1)!! atPut(d, #k2, v2)!!`."
                )
            else:
                self.stack.append(None)

    def _dispatch_int_point3d(self, x: int, yv: Any, zv: Any) -> None:
        int_cls = self.hierarchy.get("Int")
        if int_cls and int_cls.has_method("point3D"):
            m = int_cls.get_method("point3D")
            if isinstance(m, CompiledMethod):
                local_vars = [x, yv, zv, None]
                callee_frame = StackFrame(
                    instructions=m.instructions,
                    pc=0,
                    constants=m.constants or self.constants,
                    strings=m.string_table or self.string_table,
                    local_vars=local_vars,
                    receiver=x,
                    caller_frame=self.current_frame,
                    method_name="point3D",
                    class_name="Int",
                    is_block=False,
                    stack=[],
                )
                self._push_frame(callee_frame)
                if not self._in_run_loop:
                    target_depth = len(self.call_stack) - 1
                    try:
                        while len(self.call_stack) > target_depth and not self.halted:
                            f = self.call_stack[-1]
                            if f.is_exhausted:
                                self._pop_frame(f.receiver)
                                continue
                            op, arg = f.instructions[f.pc]
                            f.pc += 1
                            self._execute_opcode(op, arg)
                    except NonLocalReturn as nlr:
                        while (
                            self.call_stack
                            and self.call_stack[-1] is not nlr.home_frame
                        ):
                            popped = self.call_stack.pop()
                            popped.is_active = False
                        if self.call_stack and self.call_stack[-1] is nlr.home_frame:
                            popped = self.call_stack.pop()
                            popped.is_active = False
                        if self.call_stack:
                            self.call_stack[-1].stack.append(nlr.value)
                        else:
                            self._stack.append(nlr.value)
                        if self.call_stack and nlr.home_frame in self.call_stack:
                            raise
                return
        pt = Point3D(x=x, y=yv, z=zv)
        self.stack.append(pt)

    def _run_do_loop(self, collection: Any, block: Any) -> None:
        """Execute a 'do' iteration loop over a collection with a block closure or callable."""
        if block is None:
            return

        if isinstance(block, BlockClosure):
            for item in collection:
                if self.halted:
                    break
                if block.home_frame is not None and not block.home_frame.is_active:
                    break
                self._call_block_closure(block, [item])
        elif callable(block):
            for item in collection:
                if self.halted:
                    break
                block(item)

    def _call_block_closure(self, closure: BlockClosure, args: List[Any]) -> Any:
        """Invoke a block closure synchronously, properly handling non-local return unwinding."""
        desc = closure.descriptor
        desc_locals = getattr(desc, "locals", [])

        block_locals: List[Any] = [closure.receiver]
        block_locals.extend(args)
        block_locals.extend([None] * len(desc_locals))

        for inner_slot, cell in closure.captured_cells.items():
            while len(block_locals) <= inner_slot:
                block_locals.append(None)
            block_locals[inner_slot] = cell

        block_frame = StackFrame(
            instructions=closure.instructions,
            pc=0,
            constants=closure.constants or self.constants,
            strings=closure.string_table or self.string_table,
            local_vars=block_locals,
            receiver=closure.receiver,
            caller_frame=self.current_frame,
            method_name=getattr(desc, "name", "Block"),
            class_name=closure.home_frame.class_name if closure.home_frame else "",
            is_block=True,
            stack=[],
            home_frame=closure.home_frame,
            outer_frame=closure.outer_frame,
            closure=closure,
        )

        self._push_frame(block_frame)

        target_depth = len(self.call_stack) - 1
        while len(self.call_stack) > target_depth and not self.halted:
            f = self.call_stack[-1]
            if f.is_exhausted:
                self._pop_frame(None)
                continue
            op, arg = f.instructions[f.pc]
            f.pc += 1
            self._execute_opcode(op, arg)

        return self.stack.pop() if self.stack else None

    def _get_class_name(self, receiver: Any) -> str:
        """Extract Actor class name string from any receiver object."""
        if receiver is None:
            return "Object"
        if hasattr(receiver, "class_name"):
            cn = receiver.class_name
            return cn() if callable(cn) else str(cn)
        if isinstance(receiver, ActorList):
            return "List"
        if isinstance(receiver, ActorOrderedCollection):
            return "OrderedCollection"
        if isinstance(receiver, ActorArray):
            return "Array"
        if isinstance(receiver, list):
            return "Array"
        if isinstance(receiver, (Dictionary, ActorDictionary)):
            return "Dictionary"
        if isinstance(receiver, dict):
            return "Dictionary"
        if isinstance(receiver, Interval):
            return "Interval"
        if isinstance(receiver, Point3D):
            return "Point3D"
        if isinstance(receiver, Point):
            return "Point"
        if isinstance(receiver, Rect):
            return "Rect"
        if isinstance(receiver, bool):
            return "Boolean"
        if isinstance(receiver, int):
            return "Int"
        if isinstance(receiver, float):
            return "Float"
        if isinstance(receiver, str):
            return "String"
        return getattr(receiver, "__name__", receiver.__class__.__name__)

    def _infer_method_arity(self, receiver: Any, method_name: str) -> Optional[int]:
        """Infer parameter count for method selector from receiver or hierarchy."""
        class_name = self._get_class_name(receiver)
        if class_name and hasattr(self.hierarchy, "resolve_method"):
            m = self.hierarchy.resolve_method(class_name, method_name)
            if m is not None:
                if hasattr(m, "__actor_arity__"):
                    return getattr(m, "__actor_arity__")
                if hasattr(m, "params"):
                    return len(m.params)
                if callable(m):
                    try:
                        sig = inspect.signature(m)
                        pnames = list(sig.parameters.keys())
                        if pnames and pnames[0] in ("self", "receiver"):
                            return len(pnames) - 1
                        return len(pnames)
                    except Exception:
                        pass

        if isinstance(receiver, int):
            if method_name in ("over", "asInt", "dec", "inc", "abs", "sqrt"):
                return 1 if method_name == "over" else 0
            if method_name in ("overBy", "point3D"):
                return 2
            if method_name == "do":
                return 1
        elif isinstance(receiver, Interval):
            if method_name in ("do", "at"):
                return 1
            if method_name == "size":
                return 0
        elif isinstance(receiver, (list, tuple)):
            if method_name in ("at", "fill", "do"):
                return 1
            if method_name == "put":
                return 2
            if method_name == "size":
                return 0
        elif isinstance(receiver, (Point, Point3D)):
            if method_name in ("setX", "setY", "setZ"):
                return 1
            if method_name == "asPoint":
                return 0
            if method_name == "middlePoint":
                return 2
        elif isinstance(receiver, ActorClass):
            if method_name == "new":
                if receiver.name == "Array":
                    return 1
                return 0
        elif isinstance(receiver, BlockClosure):
            if method_name in ("eval", "value", "call"):
                return len(receiver.params)

        if receiver is None and hasattr(self.hierarchy, "_classes"):
            for cls in self.hierarchy._classes.values():
                if hasattr(cls, "has_method") and cls.has_method(method_name):
                    m = cls.get_method(method_name)
                    if hasattr(m, "params"):
                        return len(m.params)
                    if callable(m):
                        try:
                            sig = inspect.signature(m)
                            pnames = list(sig.parameters.keys())
                            if pnames and pnames[0] in ("self", "receiver"):
                                return len(pnames) - 1
                            return len(pnames)
                        except Exception:
                            pass

        return None

    def _invoke_python_callable(
        self, func: Callable, receiver: Any, args: List[Any]
    ) -> Any:
        """Invoke a Python callable adapting to its parameter signature."""
        try:
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            if param_names and param_names[0] in ("self", "receiver", "this"):
                return func(receiver, *args)
            elif len(param_names) == len(args) + 1:
                return func(receiver, *args)
            elif len(param_names) == len(args):
                return func(*args)
        except Exception:
            pass

        try:
            return func(receiver, *args)
        except TypeError:
            return func(*args)

    # =========================================================================
    # High-Level Evaluation & Dispatch Helpers
    # =========================================================================

    def _split_chunks_safely(self, source: str) -> List[str]:
        """Split Actor source on top-level `!!` delimiters outside string literals and comments."""
        chunks: List[str] = []
        buf: List[str] = []
        i = 0
        n = len(source)
        in_str: Optional[str] = None
        comment_depth = 0
        while i < n:
            ch = source[i]
            nxt = source[i + 1] if i + 1 < n else ""
            if in_str is not None:
                buf.append(ch)
                if ch == "\\" and i + 1 < n:
                    buf.append(nxt)
                    i += 2
                    continue
                if ch == in_str:
                    in_str = None
                i += 1
                continue
            if comment_depth > 0:
                buf.append(ch)
                if ch == "/" and nxt == "*":
                    buf.append(nxt)
                    comment_depth += 1
                    i += 2
                    continue
                if ch == "*" and nxt == "/":
                    buf.append(nxt)
                    comment_depth -= 1
                    i += 2
                    continue
                i += 1
                continue
            if ch == "/" and nxt == "*":
                buf.append(ch)
                buf.append(nxt)
                comment_depth += 1
                i += 2
                continue
            if ch in ('"', "'"):
                in_str = ch
                buf.append(ch)
                i += 1
                continue
            if ch == "!" and nxt == "!":
                piece = "".join(buf).strip()
                if piece:
                    chunks.append(piece)
                buf = []
                i += 2
                while i < n and source[i] == "!":
                    i += 1
                continue
            buf.append(ch)
            i += 1
        tail = "".join(buf).strip()
        if tail:
            chunks.append(tail)
        return chunks

    def eval_chunk(self, source: str) -> Any:
        """Evaluate a source string containing one or more chunks ending in !!"""
        from actorii.parser import (
            parse_actor,
            compile_to_bytecode,
            ClassDefNode,
            MethodDefNode,
            NowNode,
        )

        reentrant = bool(self.call_stack)
        saved_state = None
        if reentrant:
            saved_state = (
                list(self.call_stack),
                self._instructions,
                self._constants,
                self._string_table,
                self._pc,
                list(self._stack),
                list(self._local_vars),
                self.halted,
                self._in_run_loop,
                self._last_popped,
                self._opcodes_executed,
            )
            self.call_stack = []
            self._in_run_loop = False

        chunks = self._split_chunks_safely(source)
        last_result = None
        total_chunk_opcodes = 0

        try:
            for chunk in chunks:
                nodes = parse_actor(chunk)
                if not nodes:
                    continue

                stmt_nodes = []
                for node in nodes:
                    if isinstance(node, ClassDefNode):
                        super_name = node.super_name or "Object"
                        cls = self.hierarchy.get(node.name)
                        if cls is None:
                            cls = ActorClass(node.name, bases=(super_name,))
                            self.hierarchy.register(cls)
                            self.global_vars[node.name] = cls
                            self.global_vars["Actor"][node.name] = cls
                        cls.ivars = list(getattr(node, "ivars", []))
                        for m in getattr(node, "methods", []):
                            cm = self.compile_method_node(m, class_name=node.name)
                            self._validate_and_attach_method(cls, cm)
                        if self.fileout_dir:
                            self._file_out_class_if_enabled(node.name)
                    elif isinstance(node, NowNode):
                        self._current_compilation_class = node.target_class
                        if not self.hierarchy.get(node.target_class):
                            cls = ActorClass(node.target_class)
                            self.hierarchy.register(cls)
                            self.global_vars[node.target_class] = cls
                            self.global_vars["Actor"][node.target_class] = cls
                    elif isinstance(node, MethodDefNode):
                        target_cls_name = getattr(node, "class_name", "")
                        if (
                            not target_cls_name or target_cls_name == "Object"
                        ) and self._current_compilation_class:
                            target_cls_name = self._current_compilation_class
                        if not target_cls_name:
                            target_cls_name = "Object"
                        cls = self.hierarchy.get(target_cls_name)
                        if cls is None:
                            cls = ActorClass(target_cls_name)
                            self.hierarchy.register(cls)
                            self.global_vars[target_cls_name] = cls
                            self.global_vars["Actor"][target_cls_name] = cls
                        cm = self.compile_method_node(
                            node, class_name=target_cls_name, raw_chunk=chunk
                        )
                        self._validate_and_attach_method(cls, cm)
                        if self.fileout_dir:
                            self._file_out_class_if_enabled(target_cls_name)
                    else:
                        stmt_nodes.append(node)

                if stmt_nodes:
                    insts, consts, strings = compile_to_bytecode(stmt_nodes)
                    if (
                        len(insts) >= 2
                        and insts[-2][0] == Opcode.POP
                        and insts[-1][0] == Opcode.RETURN
                    ):
                        insts = insts[:-2] + [(Opcode.RETURN_VALUE, 0)]
                    self.load(insts, consts, strings)
                    self.run()
                    total_chunk_opcodes += self._opcodes_executed
                    if self.stack:
                        last_result = self.stack[-1]
                    elif getattr(self, "_last_popped", None) is not None:
                        last_result = self._last_popped
        except Exception:
            if self.call_stack and getattr(self, "_last_failed_frame_info", None) is None:
                top_f = self.call_stack[-1]
                cls_obj = self.hierarchy.get(top_f.class_name) if top_f.class_name else None
                m_obj = cls_obj.get_method(top_f.method_name) if cls_obj else None
                var_names = ["self"] + list(getattr(m_obj, "params", []) or []) + list(getattr(m_obj, "locals", []) or [])
                local_map = {}
                for idx_l, lv in enumerate(top_f.local_vars):
                    if idx_l == 0:
                        continue
                    v_name = var_names[idx_l] if idx_l < len(var_names) else f"local_{idx_l}"
                    raw_v = lv.value if isinstance(lv, Cell) else lv
                    local_map[str(v_name)] = repr(raw_v)[:120]
                self._last_failed_frame_info = {
                    "failedClass": str(top_f.class_name or "<Chunk>"),
                    "failedSelector": str(top_f.method_name or "<chunk>"),
                    "bytecodeOffset": max(0, int(top_f.pc) - 1),
                    "localBindings": local_map,
                }
            raise
        finally:
            self.last_eval_opcodes = total_chunk_opcodes
            if reentrant and saved_state is not None:
                (
                    self.call_stack,
                    self._instructions,
                    self._constants,
                    self._string_table,
                    self._pc,
                    self._stack,
                    self._local_vars,
                    self.halted,
                    self._in_run_loop,
                    self._last_popped,
                    outer_opcodes,
                ) = saved_state
                self._opcodes_executed = outer_opcodes + total_chunk_opcodes

        return last_result

    def eval_expression(self, source: str) -> Any:
        """Evaluate an Actor expression or statement chunk."""
        return self.eval_chunk(source)

    def dispatch_message(
        self, receiver: Any, selector: str, args: Optional[List[Any]] = None
    ) -> Any:
        """Send a message (method invocation) to an object dynamically."""
        ActorVM._active_vm = self
        self.halted = False
        if not self._in_run_loop:
            self._opcodes_executed = 0
        arguments = list(args) if args is not None else []
        self.stack.append(receiver)
        for arg in arguments:
            self.stack.append(arg)
        desc = {"method": selector, "args": len(arguments)}
        if desc not in self.constants:
            self.constants.append(desc)
        const_idx = self.constants.index(desc)
        prev_in_run_loop = self._in_run_loop
        self._in_run_loop = False
        try:
            self._handle_call_method(const_idx)
        finally:
            self._in_run_loop = prev_in_run_loop
        return self.stack.pop() if self.stack else None

    def dispatch_action(
        self,
        widget_or_window: Any,
        selector: str = "selected",
        msg: Any = None,
        args: Optional[List[Any]] = None,
    ) -> Any:
        """Dispatch a UI event action to registered widget handler or parent window."""
        ActorVM._active_vm = self
        arguments = (
            list(args) if args is not None else ([msg] if msg is not None else [])
        )

        target = None
        if isinstance(widget_or_window, (WindowsObject, ActorObject)):
            target = widget_or_window
        else:
            norm_id = str(widget_or_window)
            if norm_id.startswith("#"):
                norm_id = norm_id[1:]
            target = self.global_vars.get(norm_id) or self.global_vars["Actor"].get(
                norm_id
            )
            if target is None:
                # Search open windows/dialogs in Actor globals
                for k in self.global_vars["Actor"].keys():
                    val = self.global_vars["Actor"][k]
                    if isinstance(val, Window):
                        child = val.find_child(widget_or_window)
                        if child is not None:
                            target = child
                            break

        if target is not None:
            norm_sel = (
                str(selector)[1:] if str(selector).startswith("#") else str(selector)
            )

            # Button click hook
            if hasattr(target, "on_click") and (
                norm_sel in ("selected", "clicked", "press")
            ):
                return target.on_click(vm=self)

            # Direct method dispatch on widget
            if hasattr(target, "has_method") and target.has_method(norm_sel):
                return self.dispatch_message(target, norm_sel, arguments)
            if hasattr(target, norm_sel) and callable(getattr(target, norm_sel)):
                return self.dispatch_message(target, norm_sel, arguments)

            # Parent window Action handler dispatch
            parent = getattr(target, "parent", None)
            if parent is not None:
                handler_name = None
                child_key = str(getattr(target, "identity", getattr(target, "id", "")))
                child_norm = child_key[1:] if child_key.startswith("#") else child_key
                if hasattr(parent, "action_handlers"):
                    handler_name = parent.action_handlers.get((child_norm, norm_sel))
                target_method = handler_name or norm_sel
                payload = (
                    msg
                    if msg is not None
                    else {
                        "source": target,
                        "action": Symbol(norm_sel),
                        "id": getattr(target, "id", getattr(target, "identity", None)),
                    }
                )
                return self.dispatch_message(parent, target_method, [payload])

        return None

    def compile_method_node(
        self, node: Any, class_name: str, raw_chunk: Optional[str] = None
    ) -> CompiledMethod:
        """Compile a MethodDefNode into a CompiledMethod object."""
        from actorii.parser import BytecodeCompiler

        compiler = BytecodeCompiler()
        compiler.current_class = class_name
        compiler.compile_method(node, class_name=class_name)
        src = getattr(node, "source", None)
        if not src and raw_chunk:
            import re
            cleaned_chunk = re.sub(r"^\s*(?:/\*.*?\*/\s*)+", "", raw_chunk, flags=re.DOTALL).strip()
            m_def = re.search(r"\b(Def|Action)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", cleaned_chunk)
            if m_def:
                src = cleaned_chunk[m_def.start() :].strip()
                if not src.endswith("!!"):
                    src = src.rstrip("!") + "!!"
        cm = CompiledMethod(
            name=node.name,
            class_name=class_name,
            params=list(node.params),
            locals=list(getattr(node, "locals", [])),
            instructions=list(compiler.instructions),
            constants=list(compiler.constants),
            string_table=list(compiler.string_table),
            receiver=getattr(node, "receiver", "self") or "self",
            is_action=getattr(node, "is_action", False),
            source=src,
        )
        return cm

    # =========================================================================
    # Guardrails, Transactional Execution & Python Bridge Helpers
    # =========================================================================

    def enable_guardrails(
        self,
        enabled: bool = True,
        *,
        protect_system_classes: bool = True,
        strict_ivars: bool = True,
        max_opcodes: Optional[int] = 100_000,
        max_allowed_tier: int = 2,
        fileout_dir: Optional[str] = None,
        custom_judge: Optional[Callable[..., Any]] = None,
    ) -> None:
        """Configure static linting, risk-tier gating, opcode gas budget, and `.act` file-out."""
        self.guardrails_enabled = bool(enabled)
        self.protect_system_classes = bool(protect_system_classes) if enabled else False
        self.strict_ivars = bool(strict_ivars)
        if max_opcodes is not None:
            self.max_opcodes = int(max_opcodes)
        self.max_allowed_tier = int(max_allowed_tier)
        if fileout_dir is not None:
            self.fileout_dir = str(fileout_dir)
        if custom_judge is not None:
            self.custom_judge = custom_judge

    def _validate_and_attach_method(
        self,
        cls: ActorClass,
        cm: CompiledMethod,
        *,
        receiver: Optional[Any] = None,
        is_synthesized: bool = False,
    ) -> Any:
        """Lint, classify risk tier, run Judge policy gate if guardrails enabled, and attach `cm` to `cls`."""
        from actorii.guardrails.judge import ActorCodeJudge
        from actorii.guardrails.linter import (
            ActorGuardrailError,
            ActorLinter,
            ProtectedClassError,
            RiskTier,
        )

        linter = ActorLinter(
            self,
            strict_ivars=self.strict_ivars if (self.guardrails_enabled or is_synthesized) else False,
            protect_system_classes=self.protect_system_classes,
        )
        report = linter.lint_method(cm, class_name=cls.name, receiver=receiver)
        setattr(cm, "risk_tier", report.risk_tier)
        setattr(cm, "lint_report", report)

        if self.guardrails_enabled or self.protect_system_classes or is_synthesized:
            for err in report.errors:
                if err.code == "SYS_PROTECTED_CLASS":
                    raise ProtectedClassError(err.message)
            if self.guardrails_enabled:
                judge = ActorCodeJudge(
                    self,
                    max_allowed_tier=RiskTier(self.max_allowed_tier),
                    custom_judge=self.custom_judge,
                )
                verdict = judge.evaluate(cm, report, is_synthesized=is_synthesized)
                self.guardrail_audit_log.append(
                    {
                        "class_name": cls.name,
                        "selector": cm.name,
                        "risk_tier": int(report.risk_tier),
                        "approved": verdict.approved,
                        "reasons": list(verdict.reasons),
                        "synthesized": is_synthesized,
                    }
                )
                if not verdict.approved:
                    raise ActorGuardrailError(
                        f"Guardrail blocked {cls.name}>>{cm.name}: {'; '.join(verdict.reasons)}"
                    )

        cls.add_method(cm.name, cm)
        return report

    def _file_out_class_if_enabled(self, class_name: str) -> Optional[str]:
        """Export `<fileout_dir>/<class_name>.act` if `self.fileout_dir` is set."""
        if not self.fileout_dir:
            return None
        from actorii.guardrails.fileout import ActorFileOutManager

        mgr = ActorFileOutManager(self.fileout_dir)
        out_path = mgr.file_out_class(self, class_name)
        return str(out_path)

    def eval_transactional(
        self,
        source: str,
        *,
        max_opcodes: Optional[int] = None,
        shadow_only: bool = False,
    ) -> Any:
        """Execute `source` inside an atomic image transaction with automatic rollback on failure.

        If `shadow_only=True`, executes in speculative dry-run mode (blocking Tier 2 bridge
        side-effects and always rolling back state after returning the result).
        """
        checkpoint = self.snapshot_to_dict()
        # Preserve live Python bridge classes and ActorBridgeObject wrappers across rollback
        saved_bridge_classes = {
            cname: cls
            for cname, cls in self.hierarchy._classes.items()
            if getattr(cls, "is_bridge_class", False)
        }
        saved_bridge_objects = {}
        saved_bridge_by_id = {}
        saved_live_objects = {}
        actor_dict = self.global_vars.get("Actor", {})
        if isinstance(actor_dict, dict):
            for k, v in actor_dict.items():
                if hasattr(v, "_bridge_specs"):
                    saved_bridge_objects[k] = (v, dict(getattr(v, "_fields", {})))
                    if hasattr(v, "obj_id"):
                        saved_bridge_by_id[str(v.obj_id)] = v
                elif isinstance(v, ActorObject) and not isinstance(v, ActorClass):
                    saved_live_objects[k] = v

        saved_call_stack = list(self.call_stack)
        saved_stack = list(self.stack)

        def _restore_all() -> None:
            self.restore_from_dict(checkpoint, clear_existing=True)
            self.call_stack = list(saved_call_stack)
            for f in self.call_stack:
                f.is_active = True
            self.stack = list(saved_stack)
            for cname, bcls in saved_bridge_classes.items():
                self.hierarchy.register(bcls)
                self.global_vars[cname] = bcls
                if isinstance(self.global_vars.get("Actor"), dict):
                    self.global_vars["Actor"][cname] = bcls
            if isinstance(self.global_vars.get("Actor"), dict):
                for k, (bobj, fields_copy) in saved_bridge_objects.items():
                    bobj._fields = dict(fields_copy)
                    target_py = getattr(bobj, "_target", None)
                    if target_py is not None and hasattr(target_py, "clear_shadow_vfs"):
                        target_py.clear_shadow_vfs()
                    self.global_vars["Actor"][k] = bobj
                for k, orig_obj in saved_live_objects.items():
                    restored_obj = self.global_vars["Actor"].get(k)
                    if isinstance(restored_obj, ActorObject):
                        new_fields = dict(restored_obj._fields)
                        for fk, fv in list(new_fields.items()):
                            if isinstance(fv, ActorObject) and str(getattr(fv, "obj_id", "")) in saved_bridge_by_id:
                                new_fields[fk] = saved_bridge_by_id[str(fv.obj_id)]
                        orig_obj._fields = new_fields
                        restored_cls = self.hierarchy.get(orig_obj.class_name)
                        if restored_cls is not None:
                            orig_obj._parent = restored_cls
                            orig_obj._methods = {
                                m_k: m_v
                                for m_k, m_v in orig_obj._methods.items()
                                if m_k in getattr(orig_obj, "_instance_methods", set())
                            }
                            orig_obj.inherit_from(restored_cls)
                        if hasattr(orig_obj, "_memory") and "memory" in orig_obj._fields:
                            orig_obj._memory = orig_obj._fields["memory"]
                        if hasattr(orig_obj, "_history") and "history" in orig_obj._fields:
                            orig_obj._history = orig_obj._fields["history"]
                        if hasattr(orig_obj, "_tools") and "tools" in orig_obj._fields:
                            orig_obj._tools = orig_obj._fields["tools"]
                        self.global_vars["Actor"][k] = orig_obj

        prev_max_opcodes = self.max_opcodes
        prev_shadow = self._in_shadow_run
        if max_opcodes is not None:
            self.max_opcodes = int(max_opcodes)
        if shadow_only:
            self._in_shadow_run = True

        self._last_failed_frame_info = None
        try:
            result = self.eval_chunk(source)
        except Exception:
            if self.call_stack and getattr(self, "_last_failed_frame_info", None) is None:
                top_f = self.call_stack[-1]
                self._last_failed_frame_info = {
                    "failedClass": str(top_f.class_name or "<Chunk>"),
                    "failedSelector": str(top_f.method_name or "<chunk>"),
                    "bytecodeOffset": max(0, int(top_f.pc) - 1),
                    "localBindings": {},
                }
            _restore_all()
            raise
        finally:
            self.max_opcodes = prev_max_opcodes
            self._in_shadow_run = prev_shadow

        if shadow_only:
            _restore_all()
        return result

    def register_bridge(self, py_cls: Any, **kwargs: Any) -> ActorClass:
        """Register a Python class as a capability-gated Actor bridge class."""
        from actorii.bridge import register_bridge_class

        return register_bridge_class(self, py_cls, **kwargs)

    def expose_object(
        self, py_obj: Any, global_key: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Expose a live Python object as a capability-gated `ActorBridgeObject` in `Actor[#key]`."""
        from actorii.bridge import expose_python_object

        return expose_python_object(self, py_obj, global_key=global_key, **kwargs)

    # =========================================================================
    # Image Snapshot & Restoration (1993 ACTOR.IMG Persistence)
    # =========================================================================

    def snapshot_to_dict(self) -> Dict[str, Any]:
        """Serialize the entire live ActorVM state into a portable snapshot dictionary."""
        from actorii.image import snapshot_to_dict

        return snapshot_to_dict(self)

    def restore_from_dict(
        self, snapshot_data: Dict[str, Any], clear_existing: bool = True
    ) -> Dict[str, Any]:
        """Restore live ActorVM state from a snapshot dictionary."""
        from actorii.image import restore_from_dict

        return restore_from_dict(self, snapshot_data, clear_existing=clear_existing)

    def save_image(self, path: str = "ACTOR.IMG") -> str:
        """Save the live ActorVM image to disk as an atomic `.img` JSON snapshot."""
        from actorii.image import save_image

        return save_image(self, path)

    def snapshot(self, path: str = "ACTOR.IMG") -> str:
        """Authentic 1993 Actor snapshot alias for save_image."""
        return self.save_image(path)

    def load_image(
        self, path: str = "ACTOR.IMG", clear_existing: bool = True
    ) -> Dict[str, Any]:
        """Load an `.img` snapshot file from disk into the live ActorVM."""
        from actorii.image import load_image

        return load_image(self, path, clear_existing=clear_existing)

    # =========================================================================
    # AI + OOP Bridge & Dynamic JIT Method Synthesis
    # =========================================================================

    def set_ai_provider(self, provider: Optional[Callable[..., str]]) -> None:
        """Register a custom AI completion callable `(prompt, system_prompt, metadata) -> str`."""
        self.ai_provider = provider

    def enable_jit_synthesis(
        self,
        enabled: bool = True,
        target: Any = None,
        use_live_llm: Optional[bool] = None,
    ) -> None:
        """Enable or disable dynamic `doesNotUnderstand` JIT method synthesis."""
        if use_live_llm is not None:
            self.use_live_llm_for_jit = bool(use_live_llm)
        if target is None:
            self.jit_synthesis_enabled = bool(enabled)
        elif isinstance(target, ActorClass):
            target.jit_synthesis = bool(enabled)
        elif isinstance(target, (str, Symbol)):
            cname = str(target)[1:] if str(target).startswith("#") else str(target)
            cls = self.hierarchy.get(cname)
            if cls is not None:
                cls.jit_synthesis = bool(enabled)
        elif isinstance(target, ActorObject):
            target._fields["autoSynthesize"] = bool(enabled)
            target.auto_synthesize = bool(enabled)

    def _is_subclass_of(self, class_name: str, ancestor_name: str) -> bool:
        """Return True if `class_name` inherits from `ancestor_name` in `self.hierarchy`."""
        curr = class_name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            if curr == ancestor_name:
                return True
            cls = self.hierarchy.get(curr)
            if cls and cls._bases:
                curr = cls._bases[0]
            else:
                break
        return False

    def _collect_class_ivars(self, class_name: str) -> List[str]:
        """Collect all declared `ivars` across `class_name` and its ancestors."""
        ivars: List[str] = []
        curr = class_name
        visited = set()
        while curr and curr not in visited:
            visited.add(curr)
            cls = self.hierarchy.get(curr)
            if cls is not None:
                for iv in getattr(cls, "ivars", []):
                    if iv not in ivars:
                        ivars.append(iv)
                curr = cls._bases[0] if cls._bases else ""
            else:
                break
        return ivars

    def _should_jit_synthesize(
        self, receiver: Any, class_name: str, method_name: str
    ) -> bool:
        """Determine whether an unknown message send should trigger JIT method synthesis."""
        if receiver is None or not method_name or method_name.startswith("_"):
            return False
        if self.jit_synthesis_enabled:
            return True
        if isinstance(receiver, ActorObject):
            if getattr(receiver, "auto_synthesize", False):
                return True
            if isinstance(getattr(receiver, "_fields", None), dict) and bool(
                receiver._fields.get("autoSynthesize", False)
            ):
                return True
        if class_name and hasattr(self.hierarchy, "get"):
            cls = self.hierarchy.get(class_name)
            if cls is not None and bool(getattr(cls, "jit_synthesis", False)):
                return True
        return False

    def _jit_synthesize_method(
        self, receiver: Any, class_name: str, method_name: str, args: List[Any]
    ) -> Optional[CompiledMethod]:
        """Synthesize, compile, and attach a `CompiledMethod` for `class_name>>method_name`."""
        from actorii.ai import ActorMethodSynthesizer

        synthesizer = ActorMethodSynthesizer(self)
        return synthesizer.synthesize_method(
            receiver=receiver,
            class_name=class_name or "Object",
            selector=method_name,
            args=args,
        )

