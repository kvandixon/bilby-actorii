# -*- coding: utf-8 -*-
"""Parser for the ActorII actor language.

Implements clean-room reverse-engineered parsing of the 1993 Whitewater Actor 4.1
language syntax, preserving backward compatibility for modern Python/ActorII deployment.

The parser transforms source text into an Abstract Syntax Tree (AST) that
can be compiled into bytecode (Opcode, Operand tuples) for execution on
the ActorVM.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
import re
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

__all__ = [
    "parse_actor",
    "compile_to_bytecode",
    "clear_parser_caches",
    "get_parser_cache_stats",
    "ASTNode",
    "ClassDef",
    "ClassDefNode",
    "NowNode",
    "MethodDef",
    "MethodDefNode",
    "PrimDefNode",
    "Expr",
    "Identifier",
    "Integer",
    "RealLiteral",
    "String",
    "CharLiteral",
    "SymbolLiteralNode",
    "ArrayLiteralNode",
    "RangeCollectionNode",
    "NilLiteral",
    "BooleanLiteral",
    "SelfLiteral",
    "BinaryOp",
    "UnaryOp",
    "AttributeAccess",
    "Assignment",
    "AssignNode",
    "IndexGetNode",
    "IndexSetNode",
    "EarlyBoundNode",
    "BlockNode",
    "CallNode",
    "ReturnNode",
    "IfNode",
    "LoopNode",
    "BlockDescriptor",
    "CapturedVar",
    "MethodRef",
    "EarlyBoundDescriptor",
    "is_authentic_1993_actor",
]


# ==============================================================================
# Abstract Syntax Tree (AST) Node Hierarchy
# ==============================================================================


class ASTNode:
    """Base class for all AST nodes."""

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self.__dict__ == other.__dict__


class Expr(ASTNode):
    """Base class for expression nodes."""

    pass


# ------------------------------------------------------------------------------
# Class & Scoping AST Nodes
# ------------------------------------------------------------------------------


class ClassDef(ASTNode):
    """Represents a class definition in actor source."""

    def __init__(
        self,
        name: str,
        methods: Optional[List[MethodDef]] = None,
        bases: Tuple[str, ...] = (),
        super_name: str = "Object",
        ivars: Optional[List[str]] = None,
        class_type: int = 2,
    ):
        self.name = name
        self.methods: List[MethodDef] = methods or []
        self.bases: Tuple[str, ...] = bases or ((super_name,) if super_name else ())
        self.super_name: str = super_name or (bases[0] if bases else "Object")
        self.ivars: List[str] = ivars or []
        self.class_type: int = class_type

    def __repr__(self) -> str:
        return (
            f"ClassDef(name='{self.name}', super='{self.super_name}', "
            f"ivars={self.ivars}, methods={len(self.methods)})"
        )


class ClassDefNode(ClassDef):
    """Authentic 1993 dynamic class definition node (inherit)."""

    def __init__(
        self,
        name: str,
        super_name: str = "Object",
        ivars: Optional[List[str]] = None,
        class_type: int = 2,
        methods: Optional[List[MethodDef]] = None,
        bases: Optional[Tuple[str, ...]] = None,
    ):
        super().__init__(
            name=name,
            methods=methods or [],
            bases=bases or ((super_name,) if super_name else ()),
            super_name=super_name,
            ivars=ivars or [],
            class_type=class_type,
        )

    def __repr__(self) -> str:
        return (
            f"ClassDefNode(name='{self.name}', super='{self.super_name}', "
            f"ivars={self.ivars}, class_type={self.class_type}, methods={len(self.methods)})"
        )


class NowNode(ASTNode):
    """Represents active compilation scoping directive: now(ClassName)!!"""

    def __init__(self, target_class: str, is_meta: bool = False):
        self.target_class = target_class
        self.name = target_class
        self.is_meta = is_meta
        self.is_metaclass = is_meta

    def __repr__(self) -> str:
        meta_str = "class(" if self.is_meta else ""
        end_str = ")" if self.is_meta else ""
        return f"NowNode({meta_str}{self.target_class}{end_str})"


# ------------------------------------------------------------------------------
# Method, Action & Primitive AST Nodes
# ------------------------------------------------------------------------------


class MethodDef(ASTNode):
    """Represents a method definition in a class."""

    def __init__(
        self,
        name: str,
        params: Optional[List[str]] = None,
        body: Optional[List[ASTNode]] = None,
        class_name: str = "",
        receiver: str = "self",
        locals: Optional[List[str]] = None,
        is_action: bool = False,
        action_selectors: Any = None,
        is_primitive: bool = False,
        return_type: Optional[str] = None,
    ):
        self.name: str = name
        self.method_name: str = name
        self.params: List[str] = params or []
        self.body: List[ASTNode] = body or []
        self.class_name: str = class_name
        self.receiver: str = receiver
        self.locals: List[str] = locals or []
        self.is_action: bool = is_action
        self.action_selectors: Any = action_selectors
        self.is_primitive: bool = is_primitive
        self.return_type: Optional[str] = return_type

    def __repr__(self) -> str:
        cls_prefix = f"{self.class_name}>>" if self.class_name else ""
        kind = "Action" if self.is_action else ("Prim" if self.is_primitive else "Def")
        return (
            f"MethodDef({kind} {cls_prefix}{self.name}({self.receiver}, "
            f"params={self.params} | locals={self.locals}), body_len={len(self.body)})"
        )


class MethodDefNode(MethodDef):
    """Authentic 1993 method definition node."""

    def __init__(
        self,
        class_name: str = "",
        method_name: str = "",
        receiver: str = "self",
        params: Optional[List[str]] = None,
        locals: Optional[List[str]] = None,
        body: Optional[List[ASTNode]] = None,
        is_action: bool = False,
        action_selectors: Any = None,
        is_primitive: bool = False,
        return_type: Optional[str] = None,
    ):
        super().__init__(
            name=method_name,
            params=params or [],
            body=body or [],
            class_name=class_name,
            receiver=receiver,
            locals=locals or [],
            is_action=is_action,
            action_selectors=action_selectors,
            is_primitive=is_primitive,
            return_type=return_type,
        )

    def __repr__(self) -> str:
        cls_prefix = f"{self.class_name}>>" if self.class_name else ""
        kind = "Action" if self.is_action else ("Prim" if self.is_primitive else "Def")
        return (
            f"MethodDefNode({kind} {cls_prefix}{self.name}({self.receiver}, "
            f"params={self.params} | locals={self.locals}), body_len={len(self.body)})"
        )


class PrimDefNode(MethodDefNode):
    """Specialized node for primitive declarations: Prim name(self, ...):Type!!"""

    def __init__(
        self,
        class_name: str = "",
        method_name: str = "",
        receiver: str = "self",
        params: Optional[List[str]] = None,
        return_type: str = "Object",
    ):
        super().__init__(
            class_name=class_name,
            method_name=method_name,
            receiver=receiver,
            params=params or [],
            locals=[],
            body=[],
            is_action=False,
            action_selectors=None,
            is_primitive=True,
            return_type=return_type,
        )

    def __repr__(self) -> str:
        cls_prefix = f"{self.class_name}>>" if self.class_name else ""
        return f"PrimDefNode({cls_prefix}{self.name}({self.receiver}, params={self.params}):{self.return_type})"


# ------------------------------------------------------------------------------
# Statement & Control Flow AST Nodes
# ------------------------------------------------------------------------------


class ReturnNode(ASTNode):
    """Represents a method or block return: ^value."""

    def __init__(self, value: Optional[ASTNode] = None):
        self.value: Optional[ASTNode] = value

    def __repr__(self) -> str:
        return f"ReturnNode(^{self.value})"


class IfNode(ASTNode):
    """Represents a conditional statement: if test then body else orelse endif;"""

    def __init__(
        self,
        test: ASTNode,
        body: Optional[List[ASTNode]] = None,
        orelse: Optional[List[ASTNode]] = None,
    ):
        self.test: ASTNode = test
        self.body: List[ASTNode] = body or []
        self.orelse: List[ASTNode] = orelse or []

    def __repr__(self) -> str:
        return f"IfNode(test={self.test}, body_len={len(self.body)}, orelse_len={len(self.orelse)})"


class LoopNode(ASTNode):
    """Represents a loop construct: loop [pre;] while cond [begin] body; endLoop;"""

    def __init__(
        self,
        pre: Optional[List[ASTNode]] = None,
        cond: Optional[ASTNode] = None,
        body: Optional[List[ASTNode]] = None,
        test_at_end: bool = False,
    ):
        self.pre: List[ASTNode] = pre or []
        self.cond: Optional[ASTNode] = cond
        self.body: List[ASTNode] = body or []
        self.test_at_end: bool = test_at_end

    def __repr__(self) -> str:
        return (
            f"LoopNode(pre_len={len(self.pre)}, cond={self.cond}, "
            f"body_len={len(self.body)}, test_at_end={self.test_at_end})"
        )


# ------------------------------------------------------------------------------
# Expression AST Nodes
# ------------------------------------------------------------------------------


class Identifier(Expr):
    """Represents an identifier (variable or method name)."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Identifier(name='{self.name}')"


class Integer(Expr):
    """Represents an integer literal."""

    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"Integer(value={self.value})"


class RealLiteral(Expr):
    """Represents a floating-point real literal."""

    def __init__(self, value: float):
        self.value = value

    def __repr__(self) -> str:
        return f"RealLiteral(value={self.value})"


class String(Expr):
    """Represents a string literal."""

    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return f"String(value={self.value!r})"


class CharLiteral(Expr):
    """Represents a character literal."""

    def __init__(self, value: str):
        self.value = value

    def __repr__(self) -> str:
        return f"CharLiteral(value={self.value!r})"


class SymbolLiteralNode(Expr):
    """Represents an interned symbolic literal (#symbol)."""

    def __init__(self, name: str):
        self.name = name.lstrip("#")

    def __repr__(self) -> str:
        return f"SymbolLiteralNode(name='#{self.name}')"


class ArrayLiteralNode(Expr):
    """Represents a literal array: #(elem1 elem2 ...)."""

    def __init__(self, elements: Optional[List[ASTNode]] = None):
        self.elements: List[ASTNode] = elements or []

    def __repr__(self) -> str:
        return f"ArrayLiteralNode(elements={self.elements})"


class RangeCollectionNode(Expr):
    """Represents an action selector range collection: #[ 102-106 ], #[ #btn->#selected ]."""

    def __init__(self, elements: Optional[List[Any]] = None):
        self.elements: List[Any] = elements or []

    def __repr__(self) -> str:
        return f"RangeCollectionNode(elements={self.elements})"


class NilLiteral(Expr):
    """Represents nil literal."""

    def __repr__(self) -> str:
        return "NilLiteral()"


class BooleanLiteral(Expr):
    """Represents boolean true / false literal."""

    def __init__(self, value: bool):
        self.value = value

    def __repr__(self) -> str:
        return f"BooleanLiteral(value={self.value})"


class SelfLiteral(Expr):
    """Represents self keyword literal."""

    def __repr__(self) -> str:
        return "SelfLiteral()"


class BinaryOp(Expr):
    """Represents a binary operation (a + b, a - b, etc.)."""

    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def __repr__(self) -> str:
        return f"BinaryOp(left={self.left}, op='{self.op}', right={self.right})"


class UnaryOp(Expr):
    """Represents a unary operation (-x, not x)."""

    def __init__(self, op: str, operand: ASTNode):
        self.op = op
        self.operand = operand

    def __repr__(self) -> str:
        return f"UnaryOp(op='{self.op}', operand={self.operand})"


class AttributeAccess(Expr):
    """Represents an attribute/dot access (obj.field)."""

    def __init__(self, expr: ASTNode, attribute: str):
        self.expr = expr
        self.attribute = attribute

    def __repr__(self) -> str:
        return f"AttributeAccess(expr={self.expr}, attribute='{self.attribute}')"


class Assignment(Expr):
    """Represents an assignment statement (x = value, self.x = value)."""

    def __init__(self, target: Any, value: ASTNode):
        self.target = target
        self.value = value

    def __repr__(self) -> str:
        return f"Assignment(target={self.target}, value={self.value})"


class AssignNode(Assignment):
    """Authentic assignment expression: target := value."""

    def __init__(self, target: Union[str, ASTNode], value: ASTNode):
        target_node = Identifier(target) if isinstance(target, str) else target
        super().__init__(target=target_node, value=value)
        self.target = target
        self.target_node = target_node
        self.value = value

    def __repr__(self) -> str:
        return f"AssignNode(target={self.target}, value={self.value})"


class IndexGetNode(Expr):
    """Represents indexed read: target[index]."""

    def __init__(self, target: ASTNode, index: ASTNode):
        self.target = target
        self.index = index

    def __repr__(self) -> str:
        return f"IndexGetNode(target={self.target}, index={self.index})"


class IndexSetNode(Expr):
    """Represents indexed assignment: target[index] := value."""

    def __init__(self, target: ASTNode, index: ASTNode, value: ASTNode):
        self.target = target
        self.index = index
        self.value = value

    def __repr__(self) -> str:
        return f"IndexSetNode(target={self.target}, index={self.index}, value={self.value})"


class EarlyBoundNode(Expr):
    """Represents compile-time early-bound qualification: target:ClassName."""

    def __init__(self, target: ASTNode, class_name: str):
        self.target = target
        self.class_name = class_name

    def __repr__(self) -> str:
        return f"EarlyBoundNode(target={self.target}, class_name='{self.class_name}')"


class BlockNode(Expr):
    """Represents a block closure: {using(params | locals) body; } or { body; }"""

    def __init__(
        self,
        params: Optional[List[str]] = None,
        locals: Optional[List[str]] = None,
        body: Optional[List[ASTNode]] = None,
    ):
        self.params: List[str] = params or []
        self.locals: List[str] = locals or []
        self.body: List[ASTNode] = body or []

    def __repr__(self) -> str:
        return f"BlockNode(params={self.params}, locals={self.locals}, body_len={len(self.body)})"


class CallNode(Expr):
    """Represents a function or message call: func(arg1, arg2, ...)."""

    def __init__(self, func: Union[str, ASTNode], args: Optional[List[ASTNode]] = None):
        self.func = func
        self.args: List[ASTNode] = args or []

    def __repr__(self) -> str:
        return f"CallNode(func={self.func}, args={self.args})"


# ==============================================================================
# Tokenizer
# ==============================================================================


def _decode_triple_quoted_string(
    source: str, start: int, quote3: str
) -> Tuple[str, int]:
    """Decode multi-line triple-quoted string literals (`\"\"\"...\"\"\"` or `'''...'''`)."""
    i = start + 3
    n = len(source)
    chars: List[str] = []
    q_ch = quote3[0]
    while i < n:
        if source[i] == "\\" and source[i + 1 : i + 4] == quote3:
            chars.append(quote3)
            i += 4
            continue
        if source[i : i + 3] in ("\\\\n", "\\\\t", "\\\\r"):
            chars.append(source[i + 1 : i + 3])
            i += 3
            continue
        if source[i] == "\\" and i + 1 < n and source[i + 1] in ("n", "t", "r"):
            esc_ch = source[i + 1]
            # If the triple-quoted string contains NO actual multi-line newlines, decode \n/\t/\r as newline/tab/CR;
            # if it is already a multi-line triple-quoted Python source block, preserve \n/\t/\r literally unless preceded by \\
            end_q = source.find(quote3, start + 3)
            block_slice = source[start + 3 : end_q] if end_q != -1 else source[start + 3 :]
            if "\n" not in block_slice and esc_ch == "n":
                chars.append("\n")
                i += 2
                continue
        if source[i] == "\\" and i + 1 < n and source[i + 1] in ('"', "'"):
            chars.append(source[i + 1])
            i += 2
            continue
        if source[i : i + 3] == quote3:
            after = source[i + 3 :]
            rest = after.lstrip()
            if quote3 not in after or not rest or rest[0] in "),;+!]":
                return "".join(chars), i + 3
            chars.append(quote3)
            i += 3
            continue
        chars.append(source[i])
        i += 1
    raise SyntaxError(f"Unterminated triple-quoted string literal starting at {start}")


def _decode_string_literal(
    source: str, start: int, quote_char: str, is_c_string: bool = False
) -> Tuple[str, int]:
    """Decode string/char literals with standard C escape sequences."""
    chars: List[str] = []
    i = start + 1
    n = len(source)
    has_inner_escaped_quote = False
    while i < n:
        ch = source[i]
        if ch == quote_char:
            return "".join(chars), i + 1
        if ch == "\\":
            i += 1
            if i >= n:
                raise SyntaxError(f"Unterminated escape sequence at {start}")
            esc = source[i]
            if esc == quote_char:
                k = i + 1
                while k < n and source[k] in " \t":
                    k += 1
                is_dos_trailing_sep = (
                    not is_c_string
                    and not has_inner_escaped_quote
                    and f"\\{quote_char}" not in source[k:]
                    and "\\n" not in source[start:i]
                    and "\n" not in source[start:i]
                    and (k >= n or source[k] in "+,);]\r\n")
                )
                if is_dos_trailing_sep:
                    # In 1993 DOS paths ("res\" + attr): backslash before closing quote is a literal trailing backslash
                    chars.append("\\")
                else:
                    has_inner_escaped_quote = True
                    chars.append(quote_char)
                    i += 1
            elif esc == "n":
                chars.append("\n")
                i += 1
            elif esc == "r":
                chars.append("\r")
                i += 1
            elif esc == "t":
                chars.append("\t")
                i += 1
            elif esc == "b":
                chars.append("\b")
                i += 1
            elif esc == "f":
                chars.append("\f")
                i += 1
            elif esc == "a":
                chars.append("\a")
                i += 1
            elif esc == "v":
                chars.append("\v")
                i += 1
            elif esc == "\\":
                chars.append("\\")
                i += 1
            elif esc in ("x", "X"):
                i += 1
                hex_digits = []
                while (
                    i < n
                    and len(hex_digits) < 2
                    and source[i] in "0123456789abcdefABCDEF"
                ):
                    hex_digits.append(source[i])
                    i += 1
                if not hex_digits:
                    raise SyntaxError(f"Invalid hex escape \\x at position {i}")
                chars.append(chr(int("".join(hex_digits), 16)))
            elif esc in "01234567":
                oct_digits = [esc]
                i += 1
                while i < n and len(oct_digits) < 3 and source[i] in "01234567":
                    oct_digits.append(source[i])
                    i += 1
                chars.append(chr(int("".join(oct_digits), 8)))
            else:
                if not is_c_string:
                    chars.append("\\")
                chars.append(esc)
                i += 1
        else:
            chars.append(ch)
            i += 1
    raise SyntaxError(f"Unterminated string literal starting at {start}")


_OPERATOR_TWO_CHAR_SYMBOLS = frozenset(("<=", ">=", "<>", "~=", "=="))
_OPERATOR_ONE_CHAR_SYMBOLS = frozenset(("+", "-", "*", "/", "<", ">", "=", "^", "@"))

_TWO_CHAR_TOKENS: Dict[str, Tuple[str, str]] = {
    "!!": ("BANG_BANG", "!!"),
    ":=": ("ASSIGN", ":="),
    "::": ("DOUBLE_COLON", "::"),
    "->": ("ARROW", "->"),
    "==": ("EQ", "=="),
    "!=": ("NEQ", "!="),
    "<>": ("NEQ", "<>"),
    "~=": ("ID_NEQ", "~="),
    "<=": ("LE", "<="),
    ">=": ("GE", ">="),
}

_ONE_CHAR_TOKENS: Dict[str, Tuple[str, str]] = {
    "=": ("ASSIGN", "="),
    "^": ("CARET", "^"),
    "@": ("AT_SIGN", "@"),
    ";": ("SEMICOLON", ";"),
    "|": ("PIPE", "|"),
    "+": ("PLUS", "+"),
    "-": ("MINUS", "-"),
    "*": ("STAR", "*"),
    "/": ("SLASH", "/"),
    "(": ("LPAREN", "("),
    ")": ("RPAREN", ")"),
    "[": ("LBRACKET", "["),
    "]": ("RBRACKET", "]"),
    "{": ("LBRACE", "{"),
    "}": ("RBRACE", "}"),
    ",": ("COMMA", ","),
    ":": ("COLON", ":"),
    ".": ("DOT", "."),
    "<": ("LT", "<"),
    ">": ("GT", ">"),
}

_EXACT_KEYWORD_TYPES: Dict[str, str] = {
    "class": "CLASS",
    "def": "DEF",
    "Def": "DEF",
    "now": "NOW",
    "inherit": "INHERIT",
    "action": "ACTION",
    "Action": "ACTION",
    "prim": "PRIM",
    "Prim": "PRIM",
    "using": "USING",
    "self": "SELF",
    "nil": "NIL",
    "true": "TRUE",
    "false": "FALSE",
    "cand": "CAND",
    "cor": "COR",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "in": "IN",
    "return": "RETURN",
    "Call": "CALL",
    "import": "IMPORT",
    "from": "FROM",
    "pass": "PASS",
}

_LOWER_KEYWORD_TYPES: Dict[str, str] = {
    "if": "IF",
    "then": "THEN",
    "else": "ELSE",
    "endif": "ENDIF",
    "loop": "LOOP",
    "while": "WHILE",
    "begin": "BEGIN",
    "endloop": "ENDLOOP",
}

_HEX_CHARS = frozenset("0123456789abcdefABCDEF")
_OCT_CHARS = frozenset("01234567")
_WHITESPACE_CHARS = frozenset(" \t\r\x1a")
_PYTHON_CLASS_RE = re.compile(
    r"^\s*class\s+[a-zA-Z_][a-zA-Z0-9_]*(\([^)]*\))?\s*:",
    flags=re.MULTILINE,
)


def _match_operator_symbol(source: str, start: int) -> Optional[str]:
    """Match operator symbol following #, such as #+, #-, #*, #/, #<=, etc."""
    if start >= len(source):
        return None
    if start + 1 < len(source):
        two = source[start : start + 2]
        if two in _OPERATOR_TWO_CHAR_SYMBOLS:
            return two
        if two == "--":
            return None
    one = source[start]
    if one in _OPERATOR_ONE_CHAR_SYMBOLS:
        return one
    return None


def _tokenize(source: str) -> List[Tuple[str, str]]:
    """Tokenize actor language source code.

    Returns a list of (token_type, token_value) tuples.
    Handles both authentic 1993 Whitewater Actor syntax and legacy Pythonic syntax.
    """
    tokens: List[Tuple[str, str]] = []
    append_tok = tokens.append
    i = 0
    n = len(source)
    two_char_map = _TWO_CHAR_TOKENS
    one_char_map = _ONE_CHAR_TOKENS
    exact_kw_map = _EXACT_KEYWORD_TYPES
    lower_kw_map = _LOWER_KEYWORD_TYPES
    ws_chars = _WHITESPACE_CHARS

    while i < n:
        char = source[i]

        # Whitespace and DOS EOF (0x1A)
        if char in ws_chars:
            i += 1
            while i < n and source[i] in ws_chars:
                i += 1
            continue

        if char == "\n":
            append_tok(("NEWLINE", "\n"))
            i += 1
            continue

        # C-style comments: /* ... */ with nested comment support
        if char == "/" and i + 1 < n and source[i + 1] == "*":
            depth = 1
            i += 2
            while i < n and depth > 0:
                c2 = source[i : i + 2]
                if c2 == "/*":
                    depth += 1
                    i += 2
                elif c2 == "*/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth > 0:
                raise SyntaxError(f"Unterminated /* comment at index {i}")
            continue

        # Hash character (#) disambiguation
        if char == "#":
            if i + 1 < n:
                nxt = source[i + 1]
                # Literal array start: #(
                if nxt == "(":
                    append_tok(("HASH_LPAREN", "#("))
                    i += 2
                    continue

                # Literal range collection start: #[
                if nxt == "[":
                    append_tok(("HASH_LBRACKET", "#["))
                    i += 2
                    continue

                # Quoted symbol literal: #'...'
                if nxt == "'":
                    sym_val, next_i = _decode_string_literal(source, i + 1, "'")
                    append_tok(("SYMBOL", sym_val))
                    i = next_i
                    continue

                # Identifier symbol: #ident, #Class, #?hasElements, #_ivar
                if nxt.isalpha() or nxt in ("_", "?"):
                    j = i + 2
                    while j < n and (source[j].isalnum() or source[j] in ("_", "?")):
                        j += 1
                    append_tok(("SYMBOL", source[i + 1 : j]))
                    i = j
                    continue

                # Operator symbol: #+, #-, #*, #/, #<=, #>=, #<>, #~=, #==, #=, #<, #>, #^, #@
                op_sym = _match_operator_symbol(source, i + 1)
                if op_sym is not None:
                    append_tok(("SYMBOL", op_sym))
                    i += 1 + len(op_sym)
                    continue

            # Line comment (# ...) for backward compatibility
            nl_idx = source.find("\n", i)
            i = nl_idx if nl_idx != -1 else n
            continue

        # Multi-character operators
        if i + 1 < n:
            two_tok = two_char_map.get(source[i : i + 2])
            if two_tok is not None:
                append_tok(two_tok)
                i += 2
                continue

        # Single-character operators and delimiters
        one_tok = one_char_map.get(char)
        if one_tok is not None:
            append_tok(one_tok)
            i += 1
            continue

        if char == "%":
            if i + 1 < n and source[i + 1] == '"':
                str_val, next_i = _decode_string_literal(
                    source, i + 1, '"', is_c_string=True
                )
                append_tok(("STRING", str_val))
                i = next_i
                continue
            if i + 3 < n and source[i : i + 4] == "%'''":
                append_tok(("CHAR", "'"))
                i += 4
                continue
            if i + 1 < n and source[i + 1] == "'":
                char_val, next_i = _decode_string_literal(
                    source, i + 1, "'", is_c_string=True
                )
                if len(char_val) == 1:
                    append_tok(("CHAR", char_val))
                else:
                    append_tok(("STRING", char_val))
                i = next_i
                continue
            if i + 1 < n and (source[i + 1].isalpha() or source[i + 1] == "_"):
                j = i + 1
                while j < n and (source[j].isalnum() or source[j] in ("_", "?")):
                    j += 1
                append_tok(("IDENT", source[i:j]))
                i = j
                continue
            append_tok(("MOD", "%"))
            i += 1
            continue

        # Numbers
        if char.isdigit():
            j = i
            # Hex literal: 0x...
            if char == "0" and j + 1 < n and source[j + 1] in ("x", "X"):
                j += 2
                while j < n and source[j] in _HEX_CHARS:
                    j += 1
                append_tok(("INTEGER", source[i:j]))
                i = j
                continue

            while j < n and source[j].isdigit():
                j += 1

            # Floating point real: . followed by digit
            if j < n and source[j] == "." and j + 1 < n and source[j + 1].isdigit():
                j += 1
                while j < n and source[j].isdigit():
                    j += 1
                # Exponent e/E
                if j < n and source[j] in ("e", "E"):
                    j += 1
                    if j < n and source[j] in ("+", "-"):
                        j += 1
                    while j < n and source[j].isdigit():
                        j += 1
                append_tok(("REAL", source[i:j]))
                i = j
                continue

            # Long integer suffix L / l
            if j < n and source[j] in ("L", "l"):
                j += 1
                append_tok(("INTEGER", source[i:j]))
                i = j
                continue

            append_tok(("INTEGER", source[i:j]))
            i = j
            continue

        # String literals (triple-quoted or double-quoted)
        if char == '"':
            if source[i : i + 3] == '"""':
                str_val, next_i = _decode_triple_quoted_string(source, i, '"""')
                append_tok(("STRING", str_val))
                i = next_i
                continue
            str_val, next_i = _decode_string_literal(source, i, '"', is_c_string=False)
            append_tok(("STRING", str_val))
            i = next_i
            continue

        # Character / single-quoted string literals
        if char == "'":
            if source[i : i + 3] == "'''":
                str_val, next_i = _decode_triple_quoted_string(source, i, "'''")
                append_tok(("STRING", str_val))
                i = next_i
                continue
            if source[i : i + 3] == "'\\'":
                append_tok(("CHAR", "\\"))
                i += 3
                continue
            char_val, next_i = _decode_string_literal(source, i, "'", is_c_string=False)
            if len(char_val) == 1:
                append_tok(("CHAR", char_val))
            else:
                append_tok(("STRING", char_val))
            i = next_i
            continue

        # Identifiers and keywords
        if char.isalpha() or char in ("_", "?", "$"):
            j = i + 1
            while j < n and (source[j].isalnum() or source[j] in ("_", "?", "$")):
                j += 1
            word = source[i:j]
            kw_type = exact_kw_map.get(word)
            if kw_type is not None:
                append_tok((kw_type, word))
            else:
                lkw_type = lower_kw_map.get(word.lower())
                if lkw_type is not None:
                    append_tok((lkw_type, word))
                else:
                    append_tok(("IDENT", word))
            i = j
            continue

        # Unknown character - skip
        i += 1

    return tokens


# ==============================================================================
# Legacy Pythonic Parser Functions (Backward Compatibility for existing 37 tests)
# ==============================================================================


def _parse_expression(
    tokens: List[Tuple[str, str]], pos: int
) -> Tuple[Optional[Expr], int]:
    """Parse an expression starting at the given position in legacy mode."""
    if pos >= len(tokens):
        return None, pos

    token_type, token_value = tokens[pos]

    # Integer literal
    if token_type == "INTEGER":
        val = (
            int(token_value, 0)
            if token_value.startswith(("0x", "0X"))
            else int(token_value.rstrip("Ll"))
        )
        return Integer(val), pos + 1

    # String literal
    if token_type == "STRING":
        return String(token_value), pos + 1

    # Character literal
    if token_type == "CHAR":
        return CharLiteral(token_value), pos + 1

    # Identifier - could be variable name or start of attribute access
    if token_type == "IDENT":
        left: Expr = Identifier(token_value)
        pos += 1

        # Check for binary operations: a + b, a - b, a * b, a / b, a % b, ==, !=
        while pos < len(tokens) and tokens[pos][0] in (
            "PLUS",
            "MINUS",
            "STAR",
            "SLASH",
            "MOD",
            "EQ",
            "NEQ",
            "LT",
            "GT",
            "LE",
            "GE",
        ):
            op = tokens[pos][1]
            pos += 1

            if pos >= len(tokens):
                break

            right_token_type, right_token_value = tokens[pos]

            if right_token_type == "INTEGER":
                r_val = (
                    int(right_token_value, 0)
                    if right_token_value.startswith(("0x", "0X"))
                    else int(right_token_value.rstrip("Ll"))
                )
                right = Integer(r_val)
                pos += 1
            elif right_token_type == "STRING":
                right = String(right_token_value)
                pos += 1
            elif right_token_type == "IDENT":
                right = Identifier(right_token_value)
                pos += 1
            else:
                right = Identifier(right_token_value)
                pos += 1

            left = BinaryOp(left=left, op=op, right=right)

        return left, pos

    return Identifier(token_value), pos + 1


def _parse_assignment_stmt(
    tokens: List[Tuple[str, str]], pos: int
) -> Tuple[Optional[Assignment], int]:
    """Parse an assignment statement: target ASSIGN value in legacy mode."""
    target, pos = _parse_expression(tokens, pos)

    if pos < len(tokens) and tokens[pos][0] == "ASSIGN":
        pos += 1
        value, pos = _parse_expression(tokens, pos)
        return Assignment(target, value), pos

    return target, pos


def _parse_method_body(
    tokens: List[Tuple[str, str]], pos: int
) -> Tuple[List[ASTNode], int]:
    """Parse the body of a method in legacy mode."""
    body: List[ASTNode] = []

    while pos < len(tokens) and tokens[pos][0] == "NEWLINE":
        pos += 1

    while pos < len(tokens) and tokens[pos][0] not in ("DEF", "CLASS", "EOF"):
        while pos < len(tokens) and tokens[pos][0] == "NEWLINE":
            pos += 1

        if pos >= len(tokens) or tokens[pos][0] in ("DEF", "CLASS", "EOF"):
            break

        if (
            tokens[pos][0] == "IDENT"
            and pos + 1 < len(tokens)
            and tokens[pos + 1][0] == "ASSIGN"
        ):
            assign, pos = _parse_assignment_stmt(tokens, pos)
            if isinstance(assign, Assignment):
                body.append(assign)
                continue

        if tokens[pos][0] == "RETURN":
            pos += 1
            if pos < len(tokens) and tokens[pos][0] != "NEWLINE":
                expr, pos = _parse_expression(tokens, pos)
                if expr is not None:
                    body.append(expr)
        else:
            expr, pos = _parse_expression(tokens, pos)
            if expr is not None:
                body.append(expr)

    return body, pos


def _parse_method(tokens: List[Tuple[str, str]], pos: int) -> Tuple[MethodDef, int]:
    """Parse a legacy method definition."""
    if pos >= len(tokens):
        raise SyntaxError("Expected method name")

    method_name_token_type, method_name_token_value = tokens[pos]
    if method_name_token_type != "IDENT":
        raise SyntaxError(f"Expected method name, got {method_name_token_type}")

    method_name = method_name_token_value
    pos += 1

    params: List[str] = []
    if pos < len(tokens) and tokens[pos][0] == "LPAREN":
        pos += 1
        is_init = method_name == "__init__"

        if pos < len(tokens) and tokens[pos][0] in ("IDENT", "SELF"):
            if is_init:
                pos += 1
                while pos < len(tokens) and tokens[pos][0] != "RPAREN":
                    if tokens[pos][0] == "COMMA":
                        pos += 1
                        if pos < len(tokens) and tokens[pos][0] == "IDENT":
                            params.append(tokens[pos][1])
                            pos += 1
                    else:
                        pos += 1
            else:
                while pos < len(tokens) and tokens[pos][0] != "RPAREN":
                    if tokens[pos][0] == "COMMA":
                        pos += 1
                        continue
                    if tokens[pos][0] in ("IDENT", "SELF"):
                        params.append(tokens[pos][1])
                        pos += 1
                    else:
                        pos += 1

        if pos < len(tokens) and tokens[pos][0] == "RPAREN":
            pos += 1

    if pos < len(tokens) and tokens[pos][0] == "COLON":
        pos += 1

    while pos < len(tokens) and tokens[pos][0] == "NEWLINE":
        pos += 1

    body, pos = _parse_method_body(tokens, pos)
    method = MethodDef(name=method_name, params=params, body=body)
    return method, pos


def _parse_tokens(tokens: List[Tuple[str, str]]) -> List[ClassDef]:
    """Parse a token list into AST ClassDef nodes in legacy mode."""
    classes = []
    pos = 0
    token_count = len(tokens)

    while pos < token_count:
        token_type, token_value = tokens[pos]

        if token_type == "CLASS":
            pos += 1
            if pos >= token_count:
                raise SyntaxError("Expected class name after 'class' keyword")

            name_token_type, name_token_value = tokens[pos]
            if name_token_type != "IDENT":
                raise SyntaxError(f"Expected class name, got {name_token_type}")

            class_name = name_token_value
            pos += 1

            if pos < token_count and tokens[pos][0] == "COLON":
                pos += 1

            while pos < token_count and tokens[pos][0] == "NEWLINE":
                pos += 1

            methods = []
            while pos < token_count and tokens[pos][0] not in ("CLASS", "EOF"):
                if tokens[pos][0] == "DEF":
                    pos += 1
                    method, pos = _parse_method(tokens, pos)
                    methods.append(method)
                else:
                    pos += 1

            class_def = ClassDef(name=class_name, methods=methods, bases=())
            classes.append(class_def)
        else:
            pos += 1

    return classes


# ==============================================================================
# Authentic 1993 Whitewater Actor 4.1 Parser (Pratt / Recursive-Descent)
# ==============================================================================

# Operator Precedence Levels
PREC_NONE = 0
PREC_ASSIGN = 10  # :=
PREC_LOGICAL_OR = 20  # cor, or, bitOr, bitXor
PREC_LOGICAL_AND = 30  # cand, and, bitAnd
PREC_COMPARE = 40  # =, ==, <>, !=, ~=, <, <=, >, >=, in
PREC_ADDITIVE = 50  # +, -
PREC_MULTIPLICATIVE = 60  # *, /, mod, %
PREC_UNARY = 70  # -, ^, not
PREC_POSTFIX = 80  # [, :, ., (
PREC_PRIMARY = 90


def _get_lbp(tok: Tuple[str, str]) -> int:
    """Return left-binding power (precedence) for token in Pratt expression parsing."""
    tt, tv = tok
    if tt in ("ASSIGN", "COLON_EQUAL") and tv == ":=":
        return PREC_ASSIGN
    if tt == "COR" or tv in ("cor", "or", "bitOr", "bitXor"):
        return PREC_LOGICAL_OR
    if tt == "CAND" or tv in ("cand", "and", "bitAnd"):
        return PREC_LOGICAL_AND
    if tt in ("EQ", "NEQ", "ID_NEQ", "LE", "GE", "LT", "GT", "ARROW") or tv in (
        "=",
        "==",
        "<>",
        "!=",
        "~=",
        "<",
        "<=",
        ">",
        ">=",
        "in",
        "->",
    ):
        return PREC_COMPARE
    if tt in ("PLUS", "MINUS") or tv in ("+", "-"):
        return PREC_ADDITIVE
    if tt in ("STAR", "SLASH", "MOD") or tv in ("*", "/", "%", "mod"):
        return PREC_MULTIPLICATIVE
    if tt in ("LBRACKET", "COLON", "DOT", "LPAREN"):
        return PREC_POSTFIX
    return PREC_NONE


class Actor1993Parser:
    """Authentic 1993 Whitewater Actor 4.1 recursive-descent and Pratt expression parser."""

    def __init__(self, source: str):
        self.source = source
        # Strip NEWLINE tokens for whitespace-free parsing in authentic Actor syntax
        self.tokens: List[Tuple[str, str]] = [
            t for t in _tokenize(source) if t[0] != "NEWLINE"
        ]
        self.pos: int = 0
        self.current_class: str = "Object"
        self.is_meta: bool = False
        self.defined_classes: Dict[str, ClassDefNode] = {}

    def has_tokens(self) -> bool:
        return self.pos < len(self.tokens)

    def current_token(self) -> Tuple[str, str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ("EOF", "")

    def peek_token(self, offset: int = 1) -> Tuple[str, str]:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return ("EOF", "")

    def advance(self) -> Tuple[str, str]:
        tok = self.current_token()
        self.pos += 1
        return tok

    def match(self, *expected_types: str) -> Optional[Tuple[str, str]]:
        if self.current_token()[0] in expected_types:
            return self.advance()
        return None

    def consume(self, expected_type: str, err_msg: str) -> Tuple[str, str]:
        tok = self.current_token()
        if tok[0] == expected_type:
            return self.advance()
        raise SyntaxError(f"{err_msg} (got {tok} at position {self.pos})")

    # --------------------------------------------------------------------------
    # Top-Level Script / Chunk Parser
    # --------------------------------------------------------------------------

    def parse_all(self) -> List[ASTNode]:
        """Parse all top-level statements, definitions, and evaluation chunks."""
        nodes: List[ASTNode] = []

        while self.has_tokens():
            tok_type, tok_val = self.current_token()

            # Skip stray delimiters
            if tok_type in ("BANG_BANG", "SEMICOLON"):
                self.advance()
                continue
            if tok_type == "EOF":
                break

            # Top-level dynamic class inheritance
            if tok_type == "INHERIT":
                node = self.parse_inherit()
                nodes.append(node)
            # Active compilation scoping
            elif tok_type == "NOW":
                node = self.parse_now()
                nodes.append(node)
            # Method definition
            elif tok_type == "DEF":
                node = self.parse_def()
                nodes.append(node)
            # UI Action handler definition
            elif tok_type == "ACTION":
                node = self.parse_action()
                nodes.append(node)
            # Primitive declaration
            elif tok_type == "PRIM":
                node = self.parse_prim()
                nodes.append(node)
            # Control flow
            elif tok_type == "IF":
                node = self.parse_if()
                nodes.append(node)
            elif tok_type == "LOOP":
                node = self.parse_loop()
                nodes.append(node)
            elif tok_type == "CARET":
                self.advance()
                expr = self.parse_expression(PREC_NONE)
                self.match("SEMICOLON")
                self.match("BANG_BANG")
                nodes.append(ReturnNode(value=expr))
            else:
                # Expression or assignment statement at top level (e.g. Actor[#Sam] := new(Sieve)!!)
                expr = self.parse_expression(PREC_NONE)
                self.match("SEMICOLON")
                self.match("BANG_BANG")
                nodes.append(expr)

            # Consume any trailing delimiters
            while self.match("BANG_BANG", "SEMICOLON"):
                pass

        return nodes

    # --------------------------------------------------------------------------
    # Dynamic Class Inheritance: inherit(...)!!
    # --------------------------------------------------------------------------

    def parse_inherit(self) -> ClassDefNode:
        """Parse: inherit(SuperClass, #ClassName, #(ivars...), classType, nil)!!"""
        self.consume("INHERIT", "Expected 'inherit'")
        self.consume("LPAREN", "Expected '(' after 'inherit'")

        # 1. SuperClass
        super_tok = self.advance()
        if super_tok[0] == "NIL":
            super_name = "nil"
        elif super_tok[0] in ("IDENT", "SYMBOL"):
            super_name = super_tok[1].lstrip("#")
        else:
            raise SyntaxError(
                f"Expected superclass identifier in inherit(), got {super_tok}"
            )

        self.match("COMMA")

        # 2. #ClassName
        name_tok = self.advance()
        if name_tok[0] in ("SYMBOL", "IDENT"):
            class_name = name_tok[1].lstrip("#")
        else:
            raise SyntaxError(f"Expected class name in inherit(), got {name_tok}")

        self.match("COMMA")

        # 3. Instance variables #(x y z) or nil
        ivars: List[str] = []
        ivar_tok = self.current_token()

        if ivar_tok[0] == "NIL":
            self.advance()
        elif ivar_tok[0] == "HASH_LPAREN":
            self.advance()
            while self.has_tokens() and self.current_token()[0] not in (
                "RPAREN",
                "EOF",
            ):
                t = self.advance()
                if t[0] in ("IDENT", "SYMBOL"):
                    ivar = t[1].lstrip("#")
                    if ivar and ivar != ",":
                        ivars.append(ivar)
                self.match("COMMA")
            self.consume("RPAREN", "Expected ')' closing ivars array in inherit()")
        elif ivar_tok[0] == "LPAREN":
            self.advance()
            while self.has_tokens() and self.current_token()[0] not in (
                "RPAREN",
                "EOF",
            ):
                t = self.advance()
                if t[0] in ("IDENT", "SYMBOL"):
                    ivar = t[1].lstrip("#")
                    if ivar and ivar != ",":
                        ivars.append(ivar)
                self.match("COMMA")
            self.consume("RPAREN", "Expected ')' closing ivars list in inherit()")

        self.match("COMMA")

        # 4. classType (integer or nil, default 2)
        class_type = 2
        type_tok = self.current_token()
        if type_tok[0] == "INTEGER":
            class_type = int(self.advance()[1])
        elif type_tok[0] == "NIL":
            self.advance()

        self.match("COMMA")

        # 5. Reserved metadata / nil
        if self.current_token()[0] != "RPAREN":
            self.advance()

        self.consume("RPAREN", "Expected ')' closing inherit()")
        while self.match("BANG_BANG", "SEMICOLON"):
            pass

        node = ClassDefNode(
            name=class_name,
            super_name=super_name,
            ivars=ivars,
            class_type=class_type,
        )
        self.defined_classes[class_name] = node
        self.current_class = class_name
        return node

    # --------------------------------------------------------------------------
    # Active Compilation Scoping: now(...)!!
    # --------------------------------------------------------------------------

    def parse_now(self) -> NowNode:
        """Parse: now(ClassName)!! or now(class(ClassName))!!"""
        self.consume("NOW", "Expected 'now'")
        self.consume("LPAREN", "Expected '(' after 'now'")

        # Check for metaclass scoping: now(class(ClassName))
        if self.current_token()[0] == "CLASS" or (
            self.current_token()[0] == "IDENT" and self.current_token()[1] == "class"
        ):
            self.advance()
            self.consume("LPAREN", "Expected '(' after 'class' in now()")
            target_tok = self.advance()
            target_class = target_tok[1].lstrip("#")
            is_meta = True
            self.consume("RPAREN", "Expected ')' closing now(class(...))")
        else:
            target_tok = self.advance()
            target_class = target_tok[1].lstrip("#")
            is_meta = target_class.endswith("Class") and len(target_class) > 5

        self.consume("RPAREN", "Expected ')' closing now()")
        while self.match("BANG_BANG", "SEMICOLON"):
            pass

        self.current_class = target_class
        self.is_meta = is_meta

        if target_class not in self.defined_classes:
            self.defined_classes[target_class] = ClassDefNode(name=target_class)

        return NowNode(target_class=target_class, is_meta=is_meta)

    # --------------------------------------------------------------------------
    # Method Definition: Def name(self, args | locals) { ... }!!
    # --------------------------------------------------------------------------

    def parse_def(self) -> MethodDefNode:
        """Parse method definition: Def methodName(self, args | locals) { body; ^returnVal; }!!"""
        self.consume("DEF", "Expected 'Def'")

        name_tok = self.advance()
        method_name = name_tok[1].lstrip("#")

        self.consume("LPAREN", "Expected '(' after method name")

        receiver = "self"
        params: List[str] = []
        locals_list: List[str] = []

        if self.current_token()[0] != "RPAREN":
            # Receiver is first formal parameter
            if self.current_token()[0] in ("IDENT", "SELF", "SYMBOL", "CLASS"):
                receiver = self.advance()[1].lstrip("#")

            self.match("COMMA")

            # Arguments before '|'
            while self.has_tokens() and self.current_token()[0] not in (
                "PIPE",
                "RPAREN",
            ):
                if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
                    params.append(self.advance()[1].lstrip("#"))
                else:
                    self.advance()
                self.match("COMMA")

            # Local variables after '|'
            if self.match("PIPE"):
                while self.has_tokens() and self.current_token()[0] != "RPAREN":
                    if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
                        locals_list.append(self.advance()[1].lstrip("#"))
                    else:
                        self.advance()
                    self.match("COMMA")

        self.consume("RPAREN", "Expected ')' closing method signature")
        self.match("SEMICOLON")
        self.consume("LBRACE", "Expected '{' starting method body")

        body = self.parse_statement_list(stop_tokens=("RBRACE", "EOF"))

        self.consume("RBRACE", "Expected '}' closing method body")

        # Implicit return of self if no explicit ReturnNode at end
        if not body or not isinstance(body[-1], ReturnNode):
            body.append(ReturnNode(value=Identifier("self")))

        while self.match("BANG_BANG", "SEMICOLON"):
            pass

        method_node = MethodDefNode(
            class_name=self.current_class,
            method_name=method_name,
            receiver=receiver,
            params=params,
            locals=locals_list,
            body=body,
        )

        if self.current_class in self.defined_classes:
            self.defined_classes[self.current_class].methods.append(method_node)

        return method_node

    # --------------------------------------------------------------------------
    # UI Event Action Handler: Action name(self, msg) #[selectors] { ... }!!
    # --------------------------------------------------------------------------

    def parse_action(self) -> MethodDefNode:
        """Parse: Action actionName(self, msg) #[selectors] { body }!!"""
        self.consume("ACTION", "Expected 'Action'")

        name_tok = self.advance()
        action_name = name_tok[1].lstrip("#")

        self.consume("LPAREN", "Expected '(' after Action name")

        receiver = "self"
        if self.current_token()[0] in ("IDENT", "SELF", "SYMBOL", "CLASS"):
            receiver = self.advance()[1].lstrip("#")

        self.match("COMMA")

        msg_param = "msg"
        if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
            msg_param = self.advance()[1].lstrip("#")

        locals_list: List[str] = []
        if self.match("PIPE"):
            while self.has_tokens() and self.current_token()[0] != "RPAREN":
                if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
                    locals_list.append(self.advance()[1].lstrip("#"))
                else:
                    self.advance()
                self.match("COMMA")

        self.consume("RPAREN", "Expected ')' closing Action signature")

        selectors = self.parse_range_collection()

        self.match("SEMICOLON")
        self.consume("LBRACE", "Expected '{' starting Action body")

        body = self.parse_statement_list(stop_tokens=("RBRACE", "EOF"))

        self.consume("RBRACE", "Expected '}' closing Action body")

        if not body or not isinstance(body[-1], ReturnNode):
            body.append(ReturnNode(value=Identifier("self")))

        while self.match("BANG_BANG", "SEMICOLON"):
            pass

        action_node = MethodDefNode(
            class_name=self.current_class,
            method_name=action_name,
            receiver=receiver,
            params=[msg_param],
            locals=locals_list,
            body=body,
            is_action=True,
            action_selectors=selectors,
        )

        if self.current_class in self.defined_classes:
            self.defined_classes[self.current_class].methods.append(action_node)

        return action_node

    # --------------------------------------------------------------------------
    # Primitive Declaration: Prim name(self, args):ReturnType!!
    # --------------------------------------------------------------------------

    def parse_prim(self) -> PrimDefNode:
        """Parse: Prim name(self, args):ReturnType!!"""
        self.consume("PRIM", "Expected 'Prim'")

        name_tok = self.advance()
        method_name = name_tok[1].lstrip("#")

        self.consume("LPAREN", "Expected '(' after Prim name")

        receiver = "self"
        params: List[str] = []

        if self.current_token()[0] != "RPAREN":
            if self.current_token()[0] in ("IDENT", "SELF", "SYMBOL", "CLASS"):
                receiver = self.advance()[1].lstrip("#")

            self.match("COMMA")

            while self.has_tokens() and self.current_token()[0] != "RPAREN":
                if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
                    params.append(self.advance()[1].lstrip("#"))
                else:
                    self.advance()
                self.match("COMMA")

        self.consume("RPAREN", "Expected ')' closing Prim signature")
        self.consume("COLON", "Expected ':' before Prim return type")

        type_tok = self.advance()
        return_type = type_tok[1].lstrip("#")

        while self.match("BANG_BANG", "SEMICOLON"):
            pass

        prim_node = PrimDefNode(
            class_name=self.current_class,
            method_name=method_name,
            receiver=receiver,
            params=params,
            return_type=return_type,
        )

        if self.current_class in self.defined_classes:
            self.defined_classes[self.current_class].methods.append(prim_node)

        return prim_node

    # --------------------------------------------------------------------------
    # Literal Arrays and Range Collections
    # --------------------------------------------------------------------------

    def parse_range_collection(self) -> RangeCollectionNode:
        """Parse literal range collection: #[ 102-106 ], #[ #btn->#selected ]."""
        self.consume("HASH_LBRACKET", "Expected '#['")
        elements: List[Any] = []

        while self.has_tokens() and self.current_token()[0] not in ("RBRACKET", "EOF"):
            if self.current_token()[0] == "COMMA":
                self.advance()
                continue

            tok = self.advance()
            val = tok[1]

            # Range: INTEGER - INTEGER
            if (
                tok[0] == "INTEGER"
                and self.current_token()[0] == "MINUS"
                and self.peek_token()[0] == "INTEGER"
            ):
                self.advance()  # consume MINUS
                end_tok = self.advance()
                elements.append(("range", int(val), int(end_tok[1])))
                self.match("COMMA")
                continue

            # Association: atom -> atom
            if self.current_token()[0] == "ARROW":
                self.advance()  # consume ARROW
                val2_tok = self.advance()
                elements.append(("assoc", val.lstrip("#"), val2_tok[1].lstrip("#")))
                self.match("COMMA")
                continue

            # Discrete selector atom
            if tok[0] == "INTEGER":
                elements.append(int(val))
            elif tok[0] in ("SYMBOL", "IDENT"):
                elements.append(val.lstrip("#"))
            else:
                elements.append(val)

            self.match("COMMA")

        self.consume("RBRACKET", "Expected ']' closing '#['")
        return RangeCollectionNode(elements=elements)

    def parse_literal_array(self) -> ArrayLiteralNode:
        """Parse literal array: #( item1 item2 ... )."""
        self.consume("HASH_LPAREN", "Expected '#('")
        elements: List[ASTNode] = []

        while self.has_tokens() and self.current_token()[0] not in ("RPAREN", "EOF"):
            if self.current_token()[0] == "COMMA":
                self.advance()
                continue

            tok = self.current_token()
            if tok[0] == "HASH_LPAREN":
                elements.append(self.parse_literal_array())
            elif tok[0] == "MINUS" and self.peek_token()[0] in ("INTEGER", "REAL"):
                self.advance()  # consume MINUS
                num_tok = self.advance()
                if num_tok[0] == "INTEGER":
                    val = (
                        int(num_tok[1], 0)
                        if num_tok[1].startswith(("0x", "0X"))
                        else int(num_tok[1].rstrip("Ll"))
                    )
                    elements.append(Integer(-val))
                else:
                    elements.append(RealLiteral(-float(num_tok[1])))
            elif tok[0] == "INTEGER":
                val = (
                    int(tok[1], 0)
                    if tok[1].startswith(("0x", "0X"))
                    else int(tok[1].rstrip("Ll"))
                )
                self.advance()
                elements.append(Integer(val))
            elif tok[0] == "REAL":
                elements.append(RealLiteral(float(self.advance()[1])))
            elif tok[0] == "STRING":
                elements.append(String(self.advance()[1]))
            elif tok[0] == "SYMBOL":
                elements.append(SymbolLiteralNode(name=self.advance()[1].lstrip("#")))
            elif tok[0] == "IDENT":
                elements.append(Identifier(self.advance()[1]))
            elif tok[0] == "NIL":
                self.advance()
                elements.append(NilLiteral())
            elif tok[0] == "TRUE":
                self.advance()
                elements.append(BooleanLiteral(True))
            elif tok[0] == "FALSE":
                self.advance()
                elements.append(BooleanLiteral(False))
            else:
                elements.append(Identifier(self.advance()[1]))

            self.match("COMMA")

        self.consume("RPAREN", "Expected ')' closing '#('")
        return ArrayLiteralNode(elements=elements)

    # --------------------------------------------------------------------------
    # Block Closure: {using(params | locals) statements; }
    # --------------------------------------------------------------------------

    def parse_block_closure(self) -> BlockNode:
        """Parse block closure: {using(args | locals) body; } or { body; }"""
        self.consume("LBRACE", "Expected '{'")
        params: List[str] = []
        locals_list: List[str] = []

        if self.match("USING"):
            self.consume("LPAREN", "Expected '(' after 'using'")

            while self.has_tokens() and self.current_token()[0] not in (
                "PIPE",
                "RPAREN",
            ):
                if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
                    params.append(self.advance()[1].lstrip("#"))
                else:
                    self.advance()
                self.match("COMMA")

            if self.match("PIPE"):
                while self.has_tokens() and self.current_token()[0] != "RPAREN":
                    if self.current_token()[0] in ("IDENT", "SYMBOL", "CLASS"):
                        locals_list.append(self.advance()[1].lstrip("#"))
                    else:
                        self.advance()
                    self.match("COMMA")

            self.consume("RPAREN", "Expected ')' closing using(...)")

        body = self.parse_statement_list(stop_tokens=("RBRACE", "EOF"))
        self.consume("RBRACE", "Expected '}' closing block closure")

        return BlockNode(params=params, locals=locals_list, body=body)

    # --------------------------------------------------------------------------
    # Control Flow Statements
    # --------------------------------------------------------------------------

    def parse_statement_list(self, stop_tokens: Tuple[str, ...]) -> List[ASTNode]:
        """Parse statements separated by semicolons until a stop token is encountered."""
        stmts: List[ASTNode] = []

        while (
            self.has_tokens()
            and self.current_token()[0] not in stop_tokens
            and self.current_token()[0] != "EOF"
        ):
            if self.current_token()[0] == "SEMICOLON":
                self.advance()
                continue

            tt = self.current_token()[0]

            if tt == "IF":
                stmts.append(self.parse_if())
            elif tt == "LOOP":
                stmts.append(self.parse_loop())
            elif tt == "CARET":
                self.advance()
                if (
                    self.current_token()[0] not in stop_tokens
                    and self.current_token()[0] != "SEMICOLON"
                ):
                    expr = self.parse_expression(PREC_NONE)
                    stmts.append(ReturnNode(value=expr))
                else:
                    stmts.append(ReturnNode(value=None))
                self.match("SEMICOLON")
            elif tt == "RETURN":
                self.advance()
                if (
                    self.current_token()[0] not in stop_tokens
                    and self.current_token()[0] != "SEMICOLON"
                ):
                    expr = self.parse_expression(PREC_NONE)
                    stmts.append(ReturnNode(value=expr))
                else:
                    stmts.append(ReturnNode(value=None))
                self.match("SEMICOLON")
            else:
                expr = self.parse_expression(PREC_NONE)
                stmts.append(expr)
                self.match("SEMICOLON")

        return stmts

    def parse_if(self) -> IfNode:
        """Parse: if cond then stmts [else stmts] endif;"""
        self.consume("IF", "Expected 'if'")
        test = self.parse_expression(PREC_NONE)
        self.consume("THEN", "Expected 'then' after if condition")

        body = self.parse_statement_list(stop_tokens=("ELSE", "ENDIF", "EOF"))
        orelse: List[ASTNode] = []

        if self.match("ELSE"):
            orelse = self.parse_statement_list(stop_tokens=("ENDIF", "EOF"))

        self.consume("ENDIF", "Expected 'endif' terminating if statement")
        self.match("SEMICOLON")

        return IfNode(test=test, body=body, orelse=orelse)

    def parse_loop(self) -> LoopNode:
        """Parse: loop [pre;] while cond [begin] body; endLoop;"""
        self.consume("LOOP", "Expected 'loop'")
        pre_statements: List[ASTNode] = []
        body_statements: List[ASTNode] = []
        cond_expr: Optional[ASTNode] = None
        test_at_end = False

        if self.match("WHILE"):
            cond_expr = self.parse_expression(PREC_NONE)
            self.match("BEGIN")
            body_statements = self.parse_statement_list(stop_tokens=("ENDLOOP", "EOF"))
        else:
            stmts = self.parse_statement_list(stop_tokens=("WHILE", "ENDLOOP", "EOF"))
            if self.match("WHILE"):
                cond_expr = self.parse_expression(PREC_NONE)
                if self.match("BEGIN"):
                    pre_statements = stmts
                    body_statements = self.parse_statement_list(
                        stop_tokens=("ENDLOOP", "EOF")
                    )
                elif self.current_token()[0] == "ENDLOOP":
                    test_at_end = True
                    body_statements = stmts
                else:
                    pre_statements = stmts
                    body_statements = self.parse_statement_list(
                        stop_tokens=("ENDLOOP", "EOF")
                    )
            else:
                raise SyntaxError("Expected 'while' in loop construct")

        self.consume("ENDLOOP", "Expected 'endLoop' terminating loop")
        self.match("SEMICOLON")

        return LoopNode(
            pre=pre_statements,
            cond=cond_expr,
            body=body_statements,
            test_at_end=test_at_end,
        )

    # --------------------------------------------------------------------------
    # Pratt Expression Parser
    # --------------------------------------------------------------------------

    def parse_expression(self, rbp: int = PREC_NONE) -> ASTNode:
        """Pratt expression parser climbing precedence with right-binding power rbp."""
        tok = self.advance()
        left = self._nud(tok)

        while self.has_tokens() and rbp < _get_lbp(self.current_token()):
            op_tok = self.advance()
            left = self._led(op_tok, left)

        return left

    def _nud(self, tok: Tuple[str, str]) -> ASTNode:
        """Null denotation: atoms, literals, groups, block closures, unary operators."""
        tt, tv = tok

        if tt in ("IDENT", "CLASS"):
            return Identifier(tv)
        if tt == "INTEGER":
            val = int(tv, 0) if tv.startswith(("0x", "0X")) else int(tv.rstrip("Ll"))
            return Integer(val)
        if tt == "REAL":
            return RealLiteral(float(tv))
        if tt == "STRING":
            return String(tv)
        if tt == "CHAR":
            return CharLiteral(tv)
        if tt == "SYMBOL":
            return SymbolLiteralNode(name=tv.lstrip("#"))
        if tt == "HASH_LPAREN":
            self.pos -= 1
            return self.parse_literal_array()
        if tt == "HASH_LBRACKET":
            self.pos -= 1
            return self.parse_range_collection()
        if tt == "NIL":
            return NilLiteral()
        if tt == "TRUE":
            return BooleanLiteral(True)
        if tt == "FALSE":
            return BooleanLiteral(False)
        if tt == "SELF":
            return SelfLiteral()
        if tt == "LPAREN":
            expr = self.parse_expression(PREC_NONE)
            self.consume("RPAREN", "Expected ')' closing parenthesized expression")
            return expr
        if tt == "LBRACE":
            self.pos -= 1
            return self.parse_block_closure()
        if tt == "MINUS":
            operand = self.parse_expression(PREC_UNARY)
            if isinstance(operand, Integer):
                return Integer(-operand.value)
            if isinstance(operand, RealLiteral):
                return RealLiteral(-operand.value)
            return UnaryOp(op="-", operand=operand)
        if tt == "CARET":
            if self.pos < len(self.tokens) and self.tokens[self.pos][0] in (
                "SEMICOLON",
                "RBRACE",
                "BANG_BANG",
            ):
                return ReturnNode(value=None)
            val = self.parse_expression(PREC_NONE)
            return ReturnNode(value=val)
        if tt == "NOT" or tv == "not":
            operand = self.parse_expression(PREC_UNARY)
            return UnaryOp(op="not", operand=operand)
        if tt == "IF":
            self.pos -= 1
            return self.parse_if()
        if tt == "LOOP":
            self.pos -= 1
            return self.parse_loop()
        if tt == "CALL":
            target = self.parse_expression(PREC_POSTFIX)
            return CallNode(func=Identifier("Call"), args=[target])

        raise SyntaxError(
            f"Unexpected token in expression: {tok} at position {self.pos}"
        )

    def _led(self, op: Tuple[str, str], left: ASTNode) -> ASTNode:
        """Left denotation: infix binary operators, postfix calls, indexing, dot notation."""
        tt, tv = op

        # Postfix call: left(arg1, arg2, ...)
        if tt == "LPAREN":
            args: List[ASTNode] = []
            if self.current_token()[0] != "RPAREN":
                args.append(self.parse_expression(PREC_NONE))
                while self.match("COMMA"):
                    args.append(self.parse_expression(PREC_NONE))
            self.consume("RPAREN", "Expected ')' closing argument list")
            return CallNode(func=left, args=args)

        # Postfix indexing: left[index]
        if tt == "LBRACKET":
            idx_expr = self.parse_expression(PREC_NONE)
            self.consume("RBRACKET", "Expected ']' closing index expression")
            return IndexGetNode(target=left, index=idx_expr)

        # Postfix early-bound qualification: left:ClassName
        if tt == "COLON":
            class_tok = self.consume(
                "IDENT", "Expected class name after ':' for early-bound qualification"
            )
            return EarlyBoundNode(target=left, class_name=class_tok[1])

        # Postfix dot member access: left.attribute
        if tt == "DOT":
            if self.current_token()[0] in ("IDENT", "CLASS"):
                attr_tok = self.advance()
            else:
                attr_tok = self.consume("IDENT", "Expected attribute name after '.'")
            return AttributeAccess(expr=left, attribute=attr_tok[1])

        # Infix assignment: left := right
        if tt in ("ASSIGN", "COLON_EQUAL") and tv == ":=":
            right = self.parse_expression(PREC_ASSIGN - 1)
            if isinstance(left, IndexGetNode):
                return IndexSetNode(target=left.target, index=left.index, value=right)
            elif isinstance(left, AttributeAccess):
                return AssignNode(target=left, value=right)
            elif isinstance(left, Identifier):
                return AssignNode(target=left.name, value=right)
            else:
                return AssignNode(target=left, value=right)

        # Infix binary operations (+, -, *, /, cand, cor, =, ==, <>, ~=, etc.)
        rbp = _get_lbp(op)
        right = self.parse_expression(rbp)
        return BinaryOp(left=left, op=tv, right=right)


def _parse_actor_1993(source: str) -> List[ASTNode]:
    """Parse source string as authentic 1993 Whitewater Actor 4.1."""
    parser = Actor1993Parser(source)
    return parser.parse_all()


# ==============================================================================
# Dual-Mode Discriminator & Public Entry Points
# ==============================================================================


def _strip_comments_and_strings(source: str) -> str:
    """Strip nested block comments, line comments, and string literals.

    Preserves newlines so line-anchored regex patterns (^...) remain accurate.
    Replaces commented and string-literal characters with spaces.
    Handles:
      - Nested C-style block comments: /* ... /* ... */ ... */
      - C++ line comments: // ...
      - Hash line comments: # ...
      - Triple double-quoted strings: \"\"\"...\"\"\" (single & multiline)
      - Triple single-quoted strings: '''...''' (single & multiline)
      - Double-quoted strings: "..." (single & multiline)
      - Single-quoted strings: '...' (single & multiline)
    """
    chars: List[str] = []
    i = 0
    n = len(source)

    while i < n:
        # 1. Nested block comments: /* ... */
        if source[i : i + 2] == "/*":
            depth = 1
            chars.append("  ")
            i += 2
            while i < n and depth > 0:
                if source[i : i + 2] == "/*":
                    depth += 1
                    chars.append("  ")
                    i += 2
                elif source[i : i + 2] == "*/":
                    depth -= 1
                    chars.append("  ")
                    i += 2
                elif source[i] == "\n":
                    chars.append("\n")
                    i += 1
                else:
                    chars.append(" ")
                    i += 1
            continue

        # 2. C++ style single-line comments: // ...
        if source[i : i + 2] == "//":
            chars.append("  ")
            i += 2
            while i < n and source[i] != "\n":
                chars.append(" ")
                i += 1
            continue

        # 3. Hash single-line comments: # ...
        if source[i] == "#":
            chars.append(" ")
            i += 1
            while i < n and source[i] != "\n":
                chars.append(" ")
                i += 1
            continue

        # 4. Triple double-quoted strings: """..."""
        if source[i : i + 3] == '"""':
            chars.append("   ")
            i += 3
            while i < n:
                if source[i : i + 3] == '"""':
                    chars.append("   ")
                    i += 3
                    break
                elif source[i] == "\\" and i + 1 < n:
                    if source[i + 1] == "\n":
                        chars.append(" \n")
                    else:
                        chars.append("  ")
                    i += 2
                elif source[i] == "\n":
                    chars.append("\n")
                    i += 1
                else:
                    chars.append(" ")
                    i += 1
            continue

        # 5. Triple single-quoted strings: '''...'''
        if source[i : i + 3] == "'''":
            chars.append("   ")
            i += 3
            while i < n:
                if source[i : i + 3] == "'''":
                    chars.append("   ")
                    i += 3
                    break
                elif source[i] == "\\" and i + 1 < n:
                    if source[i + 1] == "\n":
                        chars.append(" \n")
                    else:
                        chars.append("  ")
                    i += 2
                elif source[i] == "\n":
                    chars.append("\n")
                    i += 1
                else:
                    chars.append(" ")
                    i += 1
            continue

        # 6. Single or multiline double-quoted string: "..."
        if source[i] == '"':
            chars.append(" ")
            i += 1
            while i < n:
                if source[i] == '"':
                    chars.append(" ")
                    i += 1
                    break
                elif source[i] == "\\" and i + 1 < n:
                    if source[i + 1] == "\n":
                        chars.append(" \n")
                    else:
                        chars.append("  ")
                    i += 2
                elif source[i] == "\n":
                    chars.append("\n")
                    i += 1
                else:
                    chars.append(" ")
                    i += 1
            continue

        # 7. Single or multiline single-quoted string: '...'
        if source[i] == "'":
            chars.append(" ")
            i += 1
            while i < n:
                if source[i] == "'":
                    chars.append(" ")
                    i += 1
                    break
                elif source[i] == "\\" and i + 1 < n:
                    if source[i + 1] == "\n":
                        chars.append(" \n")
                    else:
                        chars.append("  ")
                    i += 2
                elif source[i] == "\n":
                    chars.append("\n")
                    i += 1
                else:
                    chars.append(" ")
                    i += 1
            continue

        # Normal character
        chars.append(source[i])
        i += 1

    return "".join(chars)


_PARSER_CACHE_LOCK = threading.RLock()
_PARSER_CACHE_MAXSIZE = 512
_AST_LRU_CACHE: "OrderedDict[str, List[ASTNode]]" = OrderedDict()
_DISCRIMINATOR_LRU_CACHE: "OrderedDict[str, bool]" = OrderedDict()
_BYTECODE_LRU_CACHE: "OrderedDict[str, Tuple[List[Tuple[int, int]], List[Any], List[str]]]" = OrderedDict()

_PARSER_CACHE_STATS: Dict[str, int] = {
    "ast_hits": 0,
    "ast_misses": 0,
    "bytecode_hits": 0,
    "bytecode_misses": 0,
    "discriminator_hits": 0,
    "discriminator_misses": 0,
}


def clear_parser_caches() -> None:
    """Clear all thread-safe parser and bytecode compilation LRU caches."""
    with _PARSER_CACHE_LOCK:
        _AST_LRU_CACHE.clear()
        _DISCRIMINATOR_LRU_CACHE.clear()
        _BYTECODE_LRU_CACHE.clear()
        for k in _PARSER_CACHE_STATS:
            _PARSER_CACHE_STATS[k] = 0


def get_parser_cache_stats() -> Dict[str, int]:
    """Return a thread-safe snapshot of parser and bytecode cache hit/miss counters."""
    with _PARSER_CACHE_LOCK:
        stats = dict(_PARSER_CACHE_STATS)
        stats["ast_cache_size"] = len(_AST_LRU_CACHE)
        stats["bytecode_cache_size"] = len(_BYTECODE_LRU_CACHE)
        stats["discriminator_cache_size"] = len(_DISCRIMINATOR_LRU_CACHE)
        return stats


def _fast_clone_top_node(node: ASTNode, memo: Optional[Dict[int, ASTNode]] = None) -> ASTNode:
    """Fast shallow clone of a top-level AST node and its container lists, preserving shared MethodDefNode identity."""
    if memo is not None:
        nid = id(node)
        existing = memo.get(nid)
        if existing is not None:
            return existing
    cloned = copy.copy(node)
    if memo is not None:
        memo[id(node)] = cloned
    if isinstance(cloned, (MethodDefNode, MethodDef)):
        cloned.params = list(cloned.params)
        cloned.locals = list(cloned.locals)
        cloned.body = list(cloned.body)
    elif isinstance(cloned, (ClassDefNode, ClassDef)):
        if getattr(cloned, "ivars", None) is not None:
            cloned.ivars = list(cloned.ivars)
        if getattr(cloned, "methods", None) is not None:
            cloned.methods = [_fast_clone_top_node(m, memo) for m in cloned.methods]
    return cloned


def _compute_ast_top_signature(nodes: List[ASTNode]) -> tuple:
    """Compute a lightweight O(top-level) signature to detect in-place AST mutations before compile."""
    return tuple(
        (
            type(n),
            getattr(n, "name", None),
            tuple(getattr(n, "params", ()) or ()) if isinstance(getattr(n, "params", None), list) else None,
            len(getattr(n, "body", ()) or ()) if isinstance(getattr(n, "body", None), list) else None,
        )
        for n in nodes
    )


def _clone_ast_nodes(
    nodes: List[ASTNode],
    source_key: Optional[str] = None,
    source_sig: Optional[tuple] = None,
) -> List[ASTNode]:
    """Return a fresh clone of top-level `nodes` so callers can mutate AST nodes without corrupting the cache."""
    memo: Dict[int, ASTNode] = {}
    cloned = [_fast_clone_top_node(n, memo) for n in nodes]
    if source_key is not None:
        out = _ASTListWithSource(cloned)
        out._source_cache_key = source_key
        out._source_signature = source_sig if source_sig is not None else _compute_ast_top_signature(cloned)
        return out
    return cloned


class _ASTListWithSource(list):
    """List subclass carrying the immutable source key and top-level signature when AST has not been mutated."""

    _source_cache_key: Optional[str] = None
    _source_signature: Optional[tuple] = None


def is_authentic_1993_actor(source: str) -> bool:
    """Determine whether source string should be parsed with the 1993 Actor grammar.

    Returns False ONLY if source defines a Python-style class (class Foo:\n def bar:).
    Otherwise returns True to enable authentic 1993 Whitewater Actor 4.1 syntax.
    """
    if "class" not in source:
        return True
    if len(source) <= 65536:
        with _PARSER_CACHE_LOCK:
            cached = _DISCRIMINATOR_LRU_CACHE.get(source)
            if cached is not None:
                _DISCRIMINATOR_LRU_CACHE.move_to_end(source)
                _PARSER_CACHE_STATS["discriminator_hits"] += 1
                return cached
            _PARSER_CACHE_STATS["discriminator_misses"] += 1

    cleaned = _strip_comments_and_strings(source)
    res = not bool(_PYTHON_CLASS_RE.search(cleaned))

    if len(source) <= 65536:
        with _PARSER_CACHE_LOCK:
            _DISCRIMINATOR_LRU_CACHE[source] = res
            _DISCRIMINATOR_LRU_CACHE.move_to_end(source)
            while len(_DISCRIMINATOR_LRU_CACHE) > _PARSER_CACHE_MAXSIZE:
                _DISCRIMINATOR_LRU_CACHE.popitem(last=False)
    return res


def parse_actor(source: str) -> List[ASTNode]:
    """Parse actor language source code into an AST list.

    Automatically routes to legacy parser for Pythonic syntax (class Foo: def bar:),
    or to the authentic 1993 recursive-descent and Pratt parser for authentic syntax.
    Uses a thread-safe bounded LRU cache returning fresh AST node clones.

    Args:
        source: String containing actor language source code.

    Returns:
        List of AST nodes representing the parsed source.
    """
    if not source or not source.strip():
        return []

    cacheable = isinstance(source, str) and len(source) <= 65536
    if cacheable:
        with _PARSER_CACHE_LOCK:
            cached_entry = _AST_LRU_CACHE.get(source)
            if cached_entry is not None:
                _AST_LRU_CACHE.move_to_end(source)
                _PARSER_CACHE_STATS["ast_hits"] += 1
                cached_nodes, cached_sig = cached_entry
                return _clone_ast_nodes(cached_nodes, source_key=source, source_sig=cached_sig)
            _PARSER_CACHE_STATS["ast_misses"] += 1

    if is_authentic_1993_actor(source):
        parsed = _parse_actor_1993(source)
    else:
        tokens = _tokenize(source)
        parsed = _parse_tokens(tokens)

    if cacheable:
        sig = _compute_ast_top_signature(parsed)
        memo_store: Dict[int, ASTNode] = {}
        stored_template = [_fast_clone_top_node(n, memo_store) for n in parsed]
        with _PARSER_CACHE_LOCK:
            _AST_LRU_CACHE[source] = (stored_template, sig)
            _AST_LRU_CACHE.move_to_end(source)
            while len(_AST_LRU_CACHE) > _PARSER_CACHE_MAXSIZE:
                _AST_LRU_CACHE.popitem(last=False)
        out = _ASTListWithSource(parsed)
        out._source_cache_key = source
        out._source_signature = sig
        return out

    return parsed


class BlockDescriptor:
    """Descriptor for compiled block closure bytecode and lexical metadata."""

    def __init__(
        self,
        name: str,
        params: List[str],
        locals: List[str],
        instructions: List[Tuple[int, int]],
        constants: List[Any],
        string_table: List[str],
        captured_vars: Optional[List[Tuple[str, int, int]]] = None,
        has_nonlocal_return: bool = False,
    ):
        self.name = name
        self.params = params
        self.locals = locals
        self.instructions = instructions
        self.constants = constants
        self.string_table = string_table
        self.captured_vars = captured_vars or []
        self.has_nonlocal_return = has_nonlocal_return

    def __repr__(self) -> str:
        return (
            f"BlockDescriptor({self.name}, params={self.params}, "
            f"locals={self.locals}, instrs={len(self.instructions)})"
        )


class EarlyBoundDescriptor:
    """Descriptor for compile-time early-bound dispatch."""

    def __init__(self, target_class: str, method: str, args: int = 0):
        self.target_class = target_class
        self.method = method
        self.args = args

    def __repr__(self) -> str:
        return f"EarlyBoundDescriptor({self.target_class}>>{self.method}, args={self.args})"


class CapturedVar(tuple):
    """3-tuple representing captured lexical variable (name, depth, outer_slot) with inner_slot attribute."""

    def __new__(
        cls, name: str, depth: int, outer_slot: int, inner_slot: Optional[int] = None
    ):
        obj = super().__new__(cls, (name, depth, outer_slot))
        obj.name = name
        obj.depth = depth
        obj.outer_slot = outer_slot
        obj.inner_slot = inner_slot if inner_slot is not None else outer_slot
        return obj


class MethodRef(str):
    """String subclass representing a method selector with attached call arity."""

    name: str
    arg_count: int

    def __new__(cls, name: str, arg_count: int = 0):
        obj = super().__new__(cls, name)
        obj.name = name
        obj.arg_count = int(arg_count)
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, MethodRef):
            return self.name == other.name and self.arg_count == other.arg_count
        if isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.arg_count))

    def __repr__(self) -> str:
        return f"MethodRef({str(self)!r}, args={self.arg_count})"


class Scope:
    """Lexical scope tracking local variables, parameters, and closures."""

    def __init__(
        self,
        name: str = "",
        is_method: bool = False,
        is_block: bool = False,
        receiver: str = "self",
        parent: Optional["Scope"] = None,
    ):
        self.name = name
        self.is_method = is_method
        self.is_block = is_block
        self.receiver = receiver
        self.parent = parent
        self.slots: Dict[str, int] = {}
        self.slot_names: List[str] = []
        self.has_explicit_return: bool = False
        self.captured: List[CapturedVar] = []
        # Child scopes maintain isolated slots; ancestor lookups use resolve()

    def define_slot(self, name: str) -> int:
        """Define a local variable or parameter slot."""
        if name in self.slots:
            return self.slots[name]
        slot = len(self.slot_names)
        self.slots[name] = slot
        self.slot_names.append(name)
        return slot

    def lookup_local(self, name: str) -> Optional[int]:
        """Look up variable slot index in immediate local scope."""
        if name in self.slots:
            return self.slots[name]
        if (self.is_method or self.is_block) and (
            name == self.receiver or name == "self"
        ):
            return 0
        return None

    def resolve(self, name: str) -> Tuple[str, int, int]:
        """Resolve identifier across scope chain.
        Returns: (kind, depth, slot)
        kind: 'LOCAL', 'LEXICAL', or 'GLOBAL'
        """
        if name in self.slots:
            return ("LOCAL", 0, self.slots[name])
        if (self.is_method or self.is_block) and (
            name == self.receiver or name == "self"
        ):
            return ("LOCAL", 0, 0)

        if self.parent is not None:
            kind, depth, slot = self.parent.resolve(name)
            if kind in ("LOCAL", "LEXICAL"):
                actual_depth = depth + 1
                if self.is_block:
                    inner_slot = self.define_slot(name)
                    entry = CapturedVar(name, actual_depth, slot, inner_slot)
                    if not any(e[0] == name for e in self.captured):
                        self.captured.append(entry)
                    return ("LEXICAL", actual_depth, inner_slot)
                else:
                    entry = CapturedVar(name, actual_depth, slot, slot)
                    if not any(e[0] == name for e in self.captured):
                        self.captured.append(entry)
                    return ("LEXICAL", actual_depth, slot)

        return ("GLOBAL", 0, -1)


class BytecodeCompiler:
    """AST to bytecode compiler for authentic 1993 Actor and legacy Pythonic ASTs."""

    def __init__(
        self,
        constants: Optional[List[Any]] = None,
        string_table: Optional[List[str]] = None,
        parent_compiler: Optional["BytecodeCompiler"] = None,
    ):
        self.instructions: List[Tuple[int, int]] = []
        self.constants: List[Any] = constants if constants is not None else []
        self.string_table: List[str] = string_table if string_table is not None else []
        self.parent_compiler = parent_compiler
        self.current_scope: Optional[Scope] = None
        self.current_class: str = ""
        self.compiled_methods: set = set()

        if parent_compiler is not None:
            self._block_counter_ref = parent_compiler._block_counter_ref
        else:
            self._block_counter_ref = [0]

    @property
    def block_counter(self) -> int:
        return self._block_counter_ref[0]

    @block_counter.setter
    def block_counter(self, val: int) -> None:
        self._block_counter_ref[0] = val

    def emit(self, opcode: int, operand: int = 0) -> int:
        idx = len(self.instructions)
        self.instructions.append((opcode, operand))
        return idx

    def add_constant(self, val: Any) -> int:
        for idx, c in enumerate(self.constants):
            if type(c) is type(val) and c == val:
                return idx
        self.constants.append(val)
        return len(self.constants) - 1

    def add_string(self, s: str) -> int:
        if s not in self.string_table:
            self.string_table.append(s)
        return self.string_table.index(s)

    def compile_class(self, node: Union[ClassDef, ClassDefNode]):
        self.current_class = node.name
        self.add_constant(node.name)
        if hasattr(node, "super_name") and node.super_name:
            self.add_constant(node.super_name)
        for method in node.methods:
            self.compiled_methods.add(id(method))
            self.compile_method(method, class_name=node.name)

    def compile_method(
        self, method: Union[MethodDef, MethodDefNode], class_name: str = ""
    ):
        from actorii.vm import Opcode

        cls_name = class_name or getattr(method, "class_name", "") or self.current_class
        if cls_name:
            self.add_constant(cls_name)
        self.add_constant(method.name)

        receiver = getattr(method, "receiver", "self") or "self"
        method_scope = Scope(
            name=method.name,
            is_method=True,
            receiver=receiver,
            parent=None,
        )

        # Slot 0: receiver / self
        method_scope.define_slot(receiver)
        method_scope.define_slot("self")

        # Slots 1..N: parameters
        for p in method.params:
            if p not in (receiver, "self"):
                method_scope.define_slot(p)

        # Slots N+1..M: declared locals
        for loc in getattr(method, "locals", []):
            method_scope.define_slot(loc)

        prev_scope = self.current_scope
        self.current_scope = method_scope

        # Compile body statements
        for stmt in method.body:
            self.compile_stmt(stmt)

        # Implicit return of self if no explicit ^ return
        if not method_scope.has_explicit_return:
            self.emit(Opcode.LOAD_LOCAL, 0)
            self.emit(Opcode.RETURN_VALUE, 0)

        self.current_scope = prev_scope

    def compile_block(self, block: BlockNode):
        from actorii.vm import Opcode

        self.block_counter += 1
        block_name = f"Block_{self.block_counter}"
        receiver = self.current_scope.receiver if self.current_scope else "self"
        block_scope = Scope(
            name=block_name,
            is_block=True,
            receiver=receiver,
            parent=self.current_scope,
        )

        # Slot 0: receiver / closure self
        block_scope.define_slot(receiver)
        if "self" not in block_scope.slots:
            block_scope.slots["self"] = 0

        # Slots 1..N: block parameters
        for p_name in block.params:
            if p_name not in (receiver, "self"):
                block_scope.define_slot(p_name)

        # Slots N+1..M: declared block locals
        for loc in block.locals:
            block_scope.define_slot(loc)

        child_compiler = BytecodeCompiler(
            constants=self.constants,
            string_table=self.string_table,
            parent_compiler=self,
        )
        child_compiler.current_scope = block_scope
        child_compiler.current_class = self.current_class

        if not block.body:
            child_compiler.emit(Opcode.PUSH_NULL, 0)
            child_compiler.emit(Opcode.RETURN_BLOCK, 0)
        else:
            for i, stmt in enumerate(block.body):
                is_last = i == len(block.body) - 1
                if isinstance(stmt, ReturnNode):
                    block_scope.has_explicit_return = True
                    if stmt.value is not None:
                        child_compiler.compile_expr(stmt.value)
                    else:
                        child_compiler.emit(Opcode.LOAD_LOCAL, 0)
                    child_compiler.emit(Opcode.RETURN, 0)
                elif is_last and isinstance(
                    stmt,
                    (
                        Expr,
                        BinaryOp,
                        CallNode,
                        IndexGetNode,
                        Identifier,
                        Integer,
                        String,
                        SymbolLiteralNode,
                        BooleanLiteral,
                        NilLiteral,
                    ),
                ):
                    child_compiler.compile_expr(stmt)
                    child_compiler.emit(Opcode.RETURN_BLOCK, 0)
                else:
                    child_compiler.compile_stmt(stmt)

            if not child_compiler.instructions or child_compiler.instructions[-1][
                0
            ] not in (Opcode.RETURN_BLOCK, Opcode.RETURN):
                child_compiler.emit(Opcode.PUSH_NULL, 0)
                child_compiler.emit(Opcode.RETURN_BLOCK, 0)

        desc = BlockDescriptor(
            name=block_name,
            params=block.params,
            locals=block.locals,
            instructions=child_compiler.instructions,
            constants=self.constants,
            string_table=self.string_table,
            captured_vars=block_scope.captured,
            has_nonlocal_return=block_scope.has_explicit_return,
        )

        desc_idx = self.add_constant(desc)
        self.emit(Opcode.MAKE_BLOCK, desc_idx)

    def compile_assign(
        self, node: Union[Assignment, AssignNode], is_expr: bool = False
    ):
        from actorii.vm import Opcode

        target = node.target
        target_name = ""

        if isinstance(target, Identifier):
            target_name = target.name
        elif isinstance(target, str):
            target_name = target
        elif isinstance(target, AttributeAccess):
            self.compile_expr(target.expr)
            self.compile_expr(node.value)
            attr_idx = self.add_constant(target.attribute)
            self.emit(Opcode.SET_FIELD, attr_idx)
            if not is_expr:
                self.emit(Opcode.POP, 0)
            return
        elif isinstance(target, IndexGetNode):
            self.compile_index_set(
                IndexSetNode(target.target, target.index, node.value), is_expr=is_expr
            )
            return
        else:
            target_name = str(target)

        self.compile_expr(node.value)

        if is_expr:
            self.emit(Opcode.DUP, 0)

        if self.current_scope is not None:
            kind, depth, slot = self.current_scope.resolve(target_name)
            if kind in ("LOCAL", "LEXICAL"):
                self.emit(Opcode.STORE_LOCAL, slot)
            elif self.current_scope.is_method:
                new_slot = self.current_scope.define_slot(target_name)
                self.emit(Opcode.STORE_LOCAL, new_slot)
            else:
                const_idx = self.add_constant(target_name)
                self.emit(Opcode.STORE_GLOBAL, const_idx)
        else:
            const_idx = self.add_constant(target_name)
            self.emit(Opcode.STORE_GLOBAL, const_idx)

    def compile_index_get(self, node: IndexGetNode):
        from actorii.vm import Opcode

        if isinstance(node.target, EarlyBoundNode):
            self.compile_expr(node.target.target)
            self.compile_expr(node.index)
            desc = EarlyBoundDescriptor(node.target.class_name, "at", 1)
            self.add_constant(desc)
            self.emit(Opcode.GET_INDEX, 0)
        else:
            self.compile_expr(node.target)
            self.compile_expr(node.index)
            self.emit(Opcode.GET_INDEX, 0)

    def compile_index_set(self, node: IndexSetNode, is_expr: bool = False):
        from actorii.vm import Opcode

        if isinstance(node.target, EarlyBoundNode):
            self.compile_expr(node.target.target)
            self.compile_expr(node.index)
            self.compile_expr(node.value)
            desc = EarlyBoundDescriptor(node.target.class_name, "put", 2)
            self.add_constant(desc)
            self.emit(Opcode.SET_INDEX, 1 if is_expr else 0)
        else:
            self.compile_expr(node.target)
            self.compile_expr(node.index)
            self.compile_expr(node.value)
            self.emit(Opcode.SET_INDEX, 1 if is_expr else 0)

    def compile_call(self, node: CallNode):
        from actorii.vm import Opcode

        # 1. Early bound method call: target:ClassName.method(args...)
        if isinstance(node.func, AttributeAccess) and isinstance(
            node.func.expr, EarlyBoundNode
        ):
            early_node = node.func.expr
            method_name = node.func.attribute
            self.compile_expr(early_node.target)
            for arg in node.args:
                self.compile_expr(arg)
            desc = EarlyBoundDescriptor(
                early_node.class_name, method_name, len(node.args)
            )
            desc_idx = self.add_constant(desc)
            self.emit(Opcode.CALL_EARLY, desc_idx)
            return

        # 2. Early bound method call: method(target:ClassName, args...)
        if node.args and isinstance(node.args[0], EarlyBoundNode):
            early_node = node.args[0]
            method_name = (
                node.func.name if isinstance(node.func, Identifier) else str(node.func)
            )
            self.compile_expr(early_node.target)
            for arg in node.args[1:]:
                self.compile_expr(arg)
            desc = EarlyBoundDescriptor(
                early_node.class_name, method_name, len(node.args) - 1
            )
            desc_idx = self.add_constant(desc)
            self.emit(Opcode.CALL_EARLY, desc_idx)
            return

        func_name = (
            node.func.name if isinstance(node.func, Identifier) else str(node.func)
        )

        # 3. Built-in primitives: max, min, sqrt, asInt, dec, inc, fill, abs
        if func_name in ("max", "min", "sqrt", "asInt", "dec", "inc", "fill", "abs"):
            for arg in node.args:
                self.compile_expr(arg)
            prim_desc = {"name": func_name, "args": len(node.args)}
            prim_idx = self.add_constant(prim_desc)
            self.emit(Opcode.INVOKE_PRIM, prim_idx)
            return

        # 4. Authentic Actor message sending: message(receiver, arg1, ...)
        if node.args:
            receiver = node.args[0]
            method_args = node.args[1:]
            self.compile_expr(receiver)
            for arg in method_args:
                self.compile_expr(arg)
            method_ref = MethodRef(func_name, len(method_args))
            method_idx = self.add_constant(method_ref)
            self.emit(Opcode.CALL_METHOD, method_idx)
        else:
            func_idx = self.add_constant(func_name)
            self.emit(Opcode.CALL_GLOBAL, func_idx)

    def compile_if(self, node: IfNode):
        from actorii.vm import Opcode

        self.compile_expr(node.test)
        jfalse_idx = len(self.instructions)
        self.emit(Opcode.JMP_IF_FALSE, 0)

        for stmt in node.body:
            self.compile_stmt(stmt)

        if node.orelse:
            jexit_idx = len(self.instructions)
            self.emit(Opcode.JMP, 0)
            self.instructions[jfalse_idx] = (
                Opcode.JMP_IF_FALSE,
                len(self.instructions),
            )
            for stmt in node.orelse:
                self.compile_stmt(stmt)
            self.instructions[jexit_idx] = (Opcode.JMP, len(self.instructions))
        else:
            self.instructions[jfalse_idx] = (
                Opcode.JMP_IF_FALSE,
                len(self.instructions),
            )

    def compile_loop(self, node: LoopNode):
        from actorii.vm import Opcode

        loop_start = len(self.instructions)
        if node.test_at_end:
            for stmt in node.body:
                self.compile_stmt(stmt)
            if node.cond is not None:
                self.compile_expr(node.cond)
                self.emit(Opcode.JMP_IF_TRUE, loop_start)
            else:
                self.emit(Opcode.JMP, loop_start)
        else:
            for stmt in node.pre:
                self.compile_stmt(stmt)
            jexit_idx = None
            if node.cond is not None:
                self.compile_expr(node.cond)
                jexit_idx = len(self.instructions)
                self.emit(Opcode.JMP_IF_FALSE, 0)
            for stmt in node.body:
                self.compile_stmt(stmt)
            self.emit(Opcode.JMP, loop_start)
            if jexit_idx is not None:
                self.instructions[jexit_idx] = (
                    Opcode.JMP_IF_FALSE,
                    len(self.instructions),
                )

    def compile_return(self, node: ReturnNode):
        from actorii.vm import Opcode

        if self.current_scope is not None:
            self.current_scope.has_explicit_return = True
        if node.value is not None:
            self.compile_expr(node.value)
        else:
            if self.current_scope and self.current_scope.is_method:
                self.emit(Opcode.LOAD_LOCAL, 0)
            else:
                self.emit(Opcode.PUSH_NULL, 0)
        self.emit(Opcode.RETURN_VALUE, 0)

    def compile_stmt(self, stmt: Any):
        from actorii.vm import Opcode

        if stmt is None:
            return
        elif isinstance(stmt, ReturnNode):
            self.compile_return(stmt)
        elif isinstance(stmt, (Assignment, AssignNode)):
            self.compile_assign(stmt, is_expr=False)
        elif isinstance(stmt, IndexSetNode):
            self.compile_index_set(stmt, is_expr=False)
        elif isinstance(stmt, IfNode):
            self.compile_if(stmt)
        elif isinstance(stmt, LoopNode):
            self.compile_loop(stmt)
        elif isinstance(stmt, CallNode):
            self.compile_call(stmt)
            self.emit(Opcode.POP, 0)
        elif isinstance(stmt, (MethodDef, MethodDefNode)):
            self.compile_method(stmt)
        elif isinstance(stmt, (ClassDef, ClassDefNode)):
            self.compile_class(stmt)
        elif isinstance(stmt, NowNode):
            self.current_class = stmt.target_class
            self.add_constant(stmt.target_class)
        else:
            self.compile_expr(stmt)
            self.emit(Opcode.POP, 0)

    def compile_expr(self, expr: Any):
        from actorii.vm import Opcode

        if expr is None:
            self.emit(Opcode.PUSH_NULL, 0)
        elif isinstance(expr, Identifier):
            name = expr.name
            if name == "self":
                self.emit(Opcode.LOAD_LOCAL, 0)
            elif name == "nil":
                self.emit(Opcode.PUSH_NULL, 0)
            elif name == "true":
                self.emit(Opcode.PUSH_INT, 1)
            elif name == "false":
                self.emit(Opcode.PUSH_INT, 0)
            elif self.current_scope is not None:
                kind, depth, slot = self.current_scope.resolve(name)
                if kind in ("LOCAL", "LEXICAL"):
                    self.emit(Opcode.LOAD_LOCAL, slot)
                else:
                    const_idx = self.add_constant(name)
                    self.emit(Opcode.LOAD_GLOBAL, const_idx)
            else:
                const_idx = self.add_constant(name)
                self.emit(Opcode.LOAD_GLOBAL, const_idx)
        elif isinstance(expr, Integer):
            self.add_constant(expr.value)
            self.emit(Opcode.PUSH_INT, expr.value)
        elif isinstance(expr, RealLiteral):
            const_idx = self.add_constant(expr.value)
            self.emit(Opcode.LOAD_CONST, const_idx)
        elif isinstance(expr, String):
            str_idx = self.add_string(expr.value)
            self.emit(Opcode.PUSH_STR, str_idx)
        elif isinstance(expr, CharLiteral):
            str_idx = self.add_string(expr.value)
            self.emit(Opcode.PUSH_STR, str_idx)
        elif isinstance(expr, SymbolLiteralNode):
            const_idx = self.add_constant(f"#{expr.name}")
            self.emit(Opcode.LOAD_CONST, const_idx)
        elif isinstance(expr, NilLiteral):
            self.emit(Opcode.PUSH_NULL, 0)
        elif isinstance(expr, BooleanLiteral):
            self.emit(Opcode.PUSH_INT, 1 if expr.value else 0)
        elif isinstance(expr, SelfLiteral):
            self.emit(Opcode.LOAD_LOCAL, 0)
        elif isinstance(expr, ArrayLiteralNode):
            for elem in expr.elements:
                self.compile_expr(elem)
            const_idx = self.add_constant(expr.elements)
            self.emit(Opcode.LOAD_CONST, const_idx)
        elif isinstance(expr, RangeCollectionNode):
            const_idx = self.add_constant(expr.elements)
            self.emit(Opcode.LOAD_CONST, const_idx)
        elif isinstance(expr, EarlyBoundNode):
            self.compile_expr(expr.target)
        elif isinstance(expr, AttributeAccess):
            if isinstance(expr.expr, EarlyBoundNode):
                self.compile_expr(expr.expr.target)
                desc = EarlyBoundDescriptor(expr.expr.class_name, expr.attribute, 0)
                desc_idx = self.add_constant(desc)
                self.emit(Opcode.CALL_EARLY, desc_idx)
            else:
                self.compile_expr(expr.expr)
                attr_idx = self.add_constant(expr.attribute)
                self.emit(Opcode.GET_FIELD, attr_idx)
        elif isinstance(expr, BinaryOp):
            op = expr.op
            if op in ("cand", "CAND", "and", "AND"):
                self.compile_expr(expr.left)
                self.emit(Opcode.DUP, 0)
                j_end = len(self.instructions)
                self.emit(Opcode.POP_JUMP_IF_FALSE, 0)
                self.emit(Opcode.POP, 0)
                self.compile_expr(expr.right)
                self.instructions[j_end] = (
                    Opcode.POP_JUMP_IF_FALSE,
                    len(self.instructions),
                )
                return
            elif op in ("cor", "COR", "or", "OR"):
                self.compile_expr(expr.left)
                self.emit(Opcode.DUP, 0)
                j_end = len(self.instructions)
                self.emit(Opcode.POP_JUMP_IF_TRUE, 0)
                self.emit(Opcode.POP, 0)
                self.compile_expr(expr.right)
                self.instructions[j_end] = (
                    Opcode.POP_JUMP_IF_TRUE,
                    len(self.instructions),
                )
                return
            self.compile_expr(expr.left)
            self.compile_expr(expr.right)
            if op in ("+", "PLUS"):
                self.emit(Opcode.BINARY_ADD, 0)
            elif op in ("-", "MINUS"):
                self.emit(Opcode.BINARY_SUB, 0)
            elif op in ("*", "STAR"):
                self.emit(Opcode.BINARY_MUL, 0)
            elif op in ("/", "SLASH"):
                self.emit(Opcode.BINARY_DIV, 0)
            elif op in ("%", "MOD", "mod", "Mod"):
                self.emit(Opcode.BINARY_MOD, 0)
            elif op in ("<", "LT"):
                self.emit(Opcode.BINARY_LT, 0)
            elif op in ("<=", "LE"):
                self.emit(Opcode.BINARY_LE, 0)
            elif op in (">", "GT"):
                self.emit(Opcode.BINARY_GT, 0)
            elif op in (">=", "GE"):
                self.emit(Opcode.BINARY_GE, 0)
            elif op in ("==", "EQ", "="):
                self.emit(Opcode.BINARY_EQ, 0)
            elif op in ("!=", "<>", "NEQ", "~=", "ID_NEQ"):
                self.emit(Opcode.BINARY_NE, 0)
            else:
                raise NotImplementedError(f"Unsupported binary operator: {op}")
        elif isinstance(expr, UnaryOp):
            self.compile_expr(expr.operand)
            if expr.op in ("-", "MINUS"):
                self.emit(Opcode.PUSH_INT, 0)
                self.emit(Opcode.BINARY_SUB, 0)
        elif isinstance(expr, IndexGetNode):
            self.compile_index_get(expr)
        elif isinstance(expr, IndexSetNode):
            self.compile_index_set(expr, is_expr=True)
        elif isinstance(expr, BlockNode):
            self.compile_block(expr)
        elif isinstance(expr, CallNode):
            self.compile_call(expr)
        elif isinstance(expr, IfNode):
            self.compile_if(expr)
        elif isinstance(expr, LoopNode):
            self.compile_loop(expr)
        elif isinstance(expr, (Assignment, AssignNode)):
            self.compile_assign(expr, is_expr=True)
        else:
            self.emit(Opcode.PUSH_INT, 0)


def _clone_bytecode_unit(
    unit: Tuple[List[Tuple[int, int]], List[Any], List[str]],
) -> Tuple[List[Tuple[int, int]], List[Any], List[str]]:
    """Return a fresh copy of (instructions, constants, string_table) so callers can mutate lists safely."""
    insts, consts, strings = unit
    new_insts = list(insts)
    new_strings = list(strings)
    new_consts: List[Any] = []
    for c in consts:
        if isinstance(c, BlockDescriptor):
            new_consts.append(
                BlockDescriptor(
                    name=c.name,
                    params=list(c.params),
                    locals=list(c.locals),
                    instructions=list(c.instructions),
                    constants=new_consts,
                    string_table=new_strings,
                    captured_vars=list(c.captured_vars),
                    has_nonlocal_return=c.has_nonlocal_return,
                )
            )
        elif isinstance(c, list):
            new_consts.append(list(c))
        elif isinstance(c, dict):
            new_consts.append(dict(c))
        else:
            new_consts.append(c)
    return new_insts, new_consts, new_strings


def compile_to_bytecode(
    ast: List[ASTNode],
) -> Tuple[List[Tuple[int, int]], List[Any], List[str]]:
    """Compile an AST into bytecode for the ActorVM.

    Returns (instructions, constants, string_table) tuple.
    Instructions are (opcode_int, operand_int) tuples matching Opcode enum values in vm.py.
    Uses a thread-safe bounded LRU cache when `ast` carries a canonical `_source_cache_key`
    and its top-level signature is unmodified, returning fresh copies of the compiled lists.
    """
    from actorii.vm import Opcode

    cache_key: Optional[str] = getattr(ast, "_source_cache_key", None)
    if cache_key is not None:
        orig_sig = getattr(ast, "_source_signature", None)
        if orig_sig is not None and _compute_ast_top_signature(ast) != orig_sig:
            cache_key = None

    if cache_key is not None and len(cache_key) <= 65536:
        with _PARSER_CACHE_LOCK:
            cached_unit = _BYTECODE_LRU_CACHE.get(cache_key)
            if cached_unit is not None:
                _BYTECODE_LRU_CACHE.move_to_end(cache_key)
                _PARSER_CACHE_STATS["bytecode_hits"] += 1
                return _clone_bytecode_unit(cached_unit)
            _PARSER_CACHE_STATS["bytecode_misses"] += 1

    compiler = BytecodeCompiler()
    for node in ast:
        if isinstance(node, (ClassDef, ClassDefNode)):
            compiler.compile_class(node)
        elif isinstance(node, (MethodDef, MethodDefNode)):
            if id(node) not in compiler.compiled_methods:
                compiler.compile_method(node)
                compiler.compiled_methods.add(id(node))
        elif isinstance(node, NowNode):
            compiler.current_class = node.target_class
            compiler.add_constant(node.target_class)
        else:
            compiler.compile_stmt(node)

    if not compiler.instructions:
        compiler.emit(Opcode.PUSH_INT, 0)
    if compiler.instructions[-1][0] not in (
        Opcode.RETURN,
        Opcode.RETURN_VALUE,
        Opcode.HALT,
    ):
        compiler.emit(Opcode.RETURN, 0)

    unit = (compiler.instructions, compiler.constants, compiler.string_table)
    if cache_key is not None and len(cache_key) <= 65536:
        with _PARSER_CACHE_LOCK:
            _BYTECODE_LRU_CACHE[cache_key] = _clone_bytecode_unit(unit)
            _BYTECODE_LRU_CACHE.move_to_end(cache_key)
            while len(_BYTECODE_LRU_CACHE) > _PARSER_CACHE_MAXSIZE:
                _BYTECODE_LRU_CACHE.popitem(last=False)

    return unit
