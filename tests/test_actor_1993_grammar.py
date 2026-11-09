"""Comprehensive unit tests for authentic 1993 Whitewater Actor 4.1 syntax and AST.

Tests cover:
- Nested C-style block comments (/* ... /* ... */ ... */)
- Symbol literals (#ident, #Class, #'quoted', #+, #-)
- Literal arrays #(1 2 3) and range collections #[102-106], #[#btn->#selected]
- String escape sequences and char literals
- Dynamic inheritance inherit(Super, #Class, #(ivars), type, nil)!! -> ClassDefNode
- Active compilation scoping now(Class)!! and now(class(Class))!! -> NowNode
- Method definitions Def name(self, args | locals) { body; }!! -> MethodDefNode
- UI Action handlers Action name(self, msg) #[selectors] { body; }!!
- Primitive declarations Prim name(self, args):Type!! -> PrimDefNode
- Block closures {using(params | locals) body; } -> BlockNode
- Array indexing coll[i], coll[i] := val -> IndexGetNode, IndexSetNode
- Early-bound qualification target:ClassName[i] -> EarlyBoundNode
- Control flow if-then-else-endif; and loop-while-begin-endLoop;
- Global dictionary Actor[#Key] := expr!!
- Authentic historical source parsing: SIEVE.ACT, SIEVE1.ACT, FRACTAL.ACT, BUTTON.CLS, OBJECT.CLS, FRACTALW.CLS
"""

import os
import pytest
from actorii.parser import (
    parse_actor,
    Actor1993Parser,
    is_authentic_1993_actor,
    ClassDefNode,
    NowNode,
    MethodDefNode,
    PrimDefNode,
    ReturnNode,
    IfNode,
    LoopNode,
    IndexGetNode,
    IndexSetNode,
    EarlyBoundNode,
    BlockNode,
    CallNode,
    AssignNode,
    SymbolLiteralNode,
    ArrayLiteralNode,
    RangeCollectionNode,
    RealLiteral,
    CharLiteral,
    NilLiteral,
    BooleanLiteral,
    SelfLiteral,
    UnaryOp,
    BinaryOp,
    Identifier,
    Integer,
    String,
    AttributeAccess,
)


class TestCommentsAndLiterals:
    """Test tokenization of comments, symbols, arrays, ranges, and escape sequences."""

    def test_nested_block_comments(self):
        source = """
        /* Outer comment
           /* Level 1 nested
              /* Level 2 nested */
              back to level 1 */
           back to outer */
        inherit(Object, #CommentTest, nil, 2, nil)!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 1
        assert isinstance(nodes[0], ClassDefNode)
        assert nodes[0].name == "CommentTest"

    def test_symbol_literals(self):
        source = """
        inherit(Object, #SymTest, nil, 2, nil)!!
        Def testSymbols(self)
        {
            a := #standardSymbol;
            b := #'quoted symbol with spaces';
            c := #+;
            d := #-;
        }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        method = nodes[1]
        assert isinstance(method, MethodDefNode)
        assigns = [s for s in method.body if isinstance(s, AssignNode)]
        assert len(assigns) == 4
        assert assigns[0].value == SymbolLiteralNode("standardSymbol")
        assert assigns[1].value == SymbolLiteralNode("quoted symbol with spaces")
        assert assigns[2].value == SymbolLiteralNode("+")
        assert assigns[3].value == SymbolLiteralNode("-")

    def test_literal_arrays(self):
        source = """
        inherit(Object, #ArrayTest, nil, 2, nil)!!
        Def getArray(self)
        {
            ^#( 1 2.5 "hello" #sym nil true false #( inner 42 ) );
        }!!
        """
        nodes = parse_actor(source)
        method = nodes[1]
        ret = method.body[0]
        assert isinstance(ret, ReturnNode)
        arr = ret.value
        assert isinstance(arr, ArrayLiteralNode)
        elems = arr.elements
        assert isinstance(elems[0], Integer) and elems[0].value == 1
        assert isinstance(elems[1], RealLiteral) and elems[1].value == 2.5
        assert isinstance(elems[2], String) and elems[2].value == "hello"
        assert isinstance(elems[3], SymbolLiteralNode) and elems[3].name == "sym"
        assert isinstance(elems[4], NilLiteral)
        assert isinstance(elems[5], BooleanLiteral) and elems[5].value is True
        assert isinstance(elems[6], BooleanLiteral) and elems[6].value is False
        assert isinstance(elems[7], ArrayLiteralNode)
        assert len(elems[7].elements) == 2

    def test_range_collections(self):
        source = """
        inherit(Object, #RangeTest, nil, 2, nil)!!
        Def getRanges(self)
        {
            r1 := #[ 102-106, 200, 300-305 ];
            r2 := #[ #btn->#selected, #menu->#action ];
        }!!
        """
        nodes = parse_actor(source)
        method = nodes[1]
        assigns = [s for s in method.body if isinstance(s, AssignNode)]
        assert len(assigns) == 2

        r1 = assigns[0].value
        assert isinstance(r1, RangeCollectionNode)
        assert ("range", 102, 106) in r1.elements
        assert 200 in r1.elements
        assert ("range", 300, 305) in r1.elements

        r2 = assigns[1].value
        assert isinstance(r2, RangeCollectionNode)
        assert ("assoc", "btn", "selected") in r2.elements
        assert ("assoc", "menu", "action") in r2.elements

    def test_string_escapes_and_chars(self):
        source = """
        inherit(Object, #EscapeTest, nil, 2, nil)!!
        Def testEscapes(self)
        {
            s := "Line 1\\nLine 2\\tTabbed\\x41Hex";
            c := '\\n';
        }!!
        """
        nodes = parse_actor(source)
        method = nodes[1]
        assigns = [s for s in method.body if isinstance(s, AssignNode)]
        assert assigns[0].value == String("Line 1\nLine 2\tTabbedAHex")
        assert assigns[1].value == CharLiteral("\n")

    def test_macro_identifiers(self):
        source = """
        inherit(Object, #MacroTest, nil, 2, nil)!!
        Def runMacro(self)
        {
            ^%IdentityDictionary(0->#selected)[id];
        }!!
        """
        nodes = parse_actor(source)
        method = nodes[1]
        ret = method.body[0]
        assert isinstance(ret, ReturnNode)
        # ^%IdentityDictionary(0->#selected)[id] is an IndexGetNode on a CallNode on %IdentityDictionary
        idx_node = ret.value
        assert isinstance(idx_node, IndexGetNode)
        assert isinstance(idx_node.target, CallNode)
        assert idx_node.target.func == Identifier("%IdentityDictionary")


class TestDynamicInheritance:
    """Test dynamic class creation inherit(Super, #Class, #(ivars), classType, nil)!!"""

    def test_inherit_with_ivars(self):
        source = "inherit(Object, #Point, #(x y), 2, nil)!!"
        nodes = parse_actor(source)
        assert len(nodes) == 1
        cls = nodes[0]
        assert isinstance(cls, ClassDefNode)
        assert cls.name == "Point"
        assert cls.super_name == "Object"
        assert cls.ivars == ["x", "y"]
        assert cls.class_type == 2

    def test_inherit_with_nil_ivars(self):
        source = "inherit(Object, #EmptyClass, nil, 2, nil)!!"
        nodes = parse_actor(source)
        assert len(nodes) == 1
        cls = nodes[0]
        assert isinstance(cls, ClassDefNode)
        assert cls.name == "EmptyClass"
        assert cls.super_name == "Object"
        assert cls.ivars == []

    def test_inherit_subclass_hierarchy(self):
        source = """
        inherit(Object, #Shape, #(origin), 2, nil)!!
        inherit(Shape, #Circle, #(radius), 2, nil)!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        assert nodes[0].name == "Shape" and nodes[0].super_name == "Object"
        assert nodes[1].name == "Circle" and nodes[1].super_name == "Shape"
        assert nodes[1].ivars == ["radius"]


class TestActiveScoping:
    """Test now(Class)!! and now(class(Class))!! compilation scoping."""

    def test_now_class_scoping(self):
        source = """
        inherit(Object, #Worker, nil, 2, nil)!!
        now(Worker)!!
        Def doWork(self) { ^42; }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 3
        now_node = nodes[1]
        assert isinstance(now_node, NowNode)
        assert now_node.target_class == "Worker"
        assert now_node.is_metaclass is False
        method = nodes[2]
        assert isinstance(method, MethodDefNode)
        assert method.class_name == "Worker"

    def test_now_metaclass_scoping(self):
        source = """
        inherit(Object, #FractalWindow, nil, 2, nil)!!
        now(class(FractalWindow))!!
        Def open(self) { ^1; }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 3
        now_node = nodes[1]
        assert isinstance(now_node, NowNode)
        assert now_node.target_class == "FractalWindow"
        assert now_node.is_metaclass is True
        method = nodes[2]
        assert isinstance(method, MethodDefNode)
        assert method.class_name == "FractalWindow"

    def test_sequential_scoping(self):
        source = """
        inherit(Object, #A, nil, 2, nil)!!
        inherit(Object, #B, nil, 2, nil)!!
        now(A)!!
        Def methodA(self) { ^1; }!!
        now(B)!!
        Def methodB(self) { ^2; }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 6
        assert isinstance(nodes[2], NowNode) and nodes[2].target_class == "A"
        assert isinstance(nodes[3], MethodDefNode) and nodes[3].class_name == "A"
        assert isinstance(nodes[4], NowNode) and nodes[4].target_class == "B"
        assert isinstance(nodes[5], MethodDefNode) and nodes[5].class_name == "B"


class TestMethodDefinitions:
    """Test Def method definitions, receivers, arguments, locals, and returns."""

    def test_method_with_receiver_args_and_locals(self):
        source = """
        inherit(Object, #Calculator, nil, 2, nil)!!
        Def compute(self, a, b | temp result)
        {
            temp := a + b;
            result := temp * 2;
            ^result;
        }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assert m.method_name == "compute"
        assert m.receiver == "self"
        assert m.params == ["a", "b"]
        assert m.locals == ["temp", "result"]
        assert len(m.body) == 3
        assert isinstance(m.body[2], ReturnNode)
        assert m.body[2].value == Identifier("result")

    def test_implicit_self_return(self):
        source = """
        inherit(Object, #Logger, nil, 2, nil)!!
        Def log(self, msg)
        {
            self.write(msg);
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assert len(m.body) == 2
        # Implicit return of self appended
        assert isinstance(m.body[-1], ReturnNode)
        assert m.body[-1].value == Identifier("self")

    def test_class_keyword_as_identifier(self):
        source = """
        inherit(Object, #MetaReflector, nil, 2, nil)!!
        Def kindOf(self, class)
        {
            ^isAncestor(class(self), class);
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assert m.params == ["class"]
        assert isinstance(m.body[0], ReturnNode)


class TestActionHandlers:
    """Test UI Action message handlers: Action name(self, msg) #[selectors] { ... }!!"""

    def test_action_handler_numeric_range(self):
        source = """
        inherit(Object, #FractalWindow, nil, 2, nil)!!
        Action command(self, msg) #[ 102-106 ]
        {
            paletteIndex := msg - 102;
            self.drawPalette(paletteIndex);
        }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        act = nodes[1]
        assert isinstance(act, MethodDefNode)
        assert act.is_action is True
        assert act.method_name == "command"
        assert act.receiver == "self"
        assert act.params == ["msg"]
        assert isinstance(act.action_selectors, RangeCollectionNode)
        assert ("range", 102, 106) in act.action_selectors.elements

    def test_action_handler_association(self):
        source = """
        inherit(Object, #ButtonWindow, nil, 2, nil)!!
        Action buttonPress(self, msg) #[ #btn->#selected ]
        {
            self.activate();
        }!!
        """
        nodes = parse_actor(source)
        act = nodes[1]
        assert isinstance(act, MethodDefNode)
        assert act.is_action is True
        assert act.action_selectors.elements == [("assoc", "btn", "selected")]


class TestPrimitives:
    """Test Prim primitive declarations: Prim name(self, args):ReturnType!!"""

    def test_prim_declarations(self):
        source = """
        inherit(Object, #PrimitiveHolder, nil, 2, nil)!!
        Prim hash(self):Int!!
        Prim printOn(self, aStream):self!!
        Prim setTag(self, tag):self!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 4
        prim1, prim2, prim3 = nodes[1], nodes[2], nodes[3]

        assert isinstance(prim1, PrimDefNode)
        assert prim1.method_name == "hash"
        assert prim1.receiver == "self"
        assert prim1.params == []
        assert prim1.return_type == "Int"

        assert isinstance(prim2, PrimDefNode)
        assert prim2.method_name == "printOn"
        assert prim2.params == ["aStream"]
        assert prim2.return_type == "self"

        assert isinstance(prim3, PrimDefNode)
        assert prim3.method_name == "setTag"
        assert prim3.params == ["tag"]
        assert prim3.return_type == "self"


class TestBlockClosures:
    """Test block closures: {using(args | locals) body; } and invocation."""

    def test_block_with_args_and_locals(self):
        source = """
        inherit(Object, #BlockUser, nil, 2, nil)!!
        Def runBlock(self)
        {
            blk := {using(item, index | squared)
                squared := item * item;
                ^squared + index;
            };
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assign = m.body[0]
        assert isinstance(assign, AssignNode)
        blk = assign.value
        assert isinstance(blk, BlockNode)
        assert blk.params == ["item", "index"]
        assert blk.locals == ["squared"]
        assert len(blk.body) == 2

    def test_block_without_using(self):
        source = """
        inherit(Object, #SimpleBlock, nil, 2, nil)!!
        Def evaluate(self)
        {
            blk := { self.render(); };
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assign = m.body[0]
        blk = assign.value
        assert isinstance(blk, BlockNode)
        assert blk.params == []
        assert blk.locals == []
        assert len(blk.body) == 1


class TestIndexingAndEarlyBound:
    """Test array/dictionary indexing coll[i] and early-bound target:Class[i]."""

    def test_index_get_and_set(self):
        source = """
        inherit(Object, #IndexTest, nil, 2, nil)!!
        Def testIndexing(self, coll, i, val)
        {
            coll[i] := val;
            ^coll[i];
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        set_node = m.body[0]
        assert isinstance(set_node, IndexSetNode)
        assert set_node.target == Identifier("coll")
        assert set_node.index == Identifier("i")
        assert set_node.value == Identifier("val")

        ret_node = m.body[1]
        assert isinstance(ret_node, ReturnNode)
        assert isinstance(ret_node.value, IndexGetNode)
        assert ret_node.value.target == Identifier("coll")
        assert ret_node.value.index == Identifier("i")

    def test_early_bound_qualification(self):
        source = """
        inherit(Object, #EarlyBoundTest, nil, 2, nil)!!
        Def runEarlyBound(self, flags, i)
        {
            flags:Object[i] := nil;
            val := flags:Object[i];
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        set_node = m.body[0]
        assert isinstance(set_node, IndexSetNode)
        assert isinstance(set_node.target, EarlyBoundNode)
        assert set_node.target.target == Identifier("flags")
        assert set_node.target.class_name == "Object"

        assign_node = m.body[1]
        assert isinstance(assign_node, AssignNode)
        get_node = assign_node.value
        assert isinstance(get_node, IndexGetNode)
        assert isinstance(get_node.target, EarlyBoundNode)
        assert get_node.target.class_name == "Object"


class TestControlFlow:
    """Test if-then-else-endif; and loop-while-begin-endLoop;."""

    def test_if_then_else_statement(self):
        source = """
        inherit(Object, #FlowTest, nil, 2, nil)!!
        Def check(self, x)
        {
            if x > 0 then
                res := 1;
            else
                res := -1;
            endif;
            ^res;
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        if_node = m.body[0]
        assert isinstance(if_node, IfNode)
        assert isinstance(if_node.test, BinaryOp)
        assert if_node.test.op == ">"
        assert len(if_node.body) == 1
        assert len(if_node.orelse) == 1

    def test_if_expression(self):
        source = """
        inherit(Object, #IfExprTest, nil, 2, nil)!!
        Def computeSign(self, x)
        {
            str := if x > 0 then "positive" else "non-positive" endif;
            ^str;
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assign_node = m.body[0]
        assert isinstance(assign_node, AssignNode)
        assert isinstance(assign_node.value, IfNode)

    def test_loop_while_begin_endloop(self):
        source = """
        inherit(Object, #LoopTest, nil, 2, nil)!!
        Def countUp(self, n | i)
        {
            i := 0;
            loop
            while i < n
            begin
                i := i + 1;
            endLoop;
            ^i;
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        loop_node = m.body[1]
        assert isinstance(loop_node, LoopNode)
        assert isinstance(loop_node.cond, BinaryOp)
        assert loop_node.cond.op == "<"
        assert len(loop_node.body) == 1

    def test_loop_with_pre_statements(self):
        source = """
        inherit(Object, #SieveLoopTest, nil, 2, nil)!!
        Def run(self, count | prime k)
        {
            loop
                prime := 3;
            while prime <= 100
            begin
                k := prime + prime;
                prime := prime + 2;
            endLoop;
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        loop_node = m.body[0]
        assert isinstance(loop_node, LoopNode)
        assert len(loop_node.pre) == 1
        assert isinstance(loop_node.pre[0], AssignNode)
        assert len(loop_node.body) == 2


class TestGlobalDictionary:
    """Test global dictionary Actor[#Key] access and top-level trigger !!."""

    def test_actor_dictionary_assignment(self):
        source = """
        inherit(Object, #Sieve, nil, 2, nil)!!
        Actor[#Sam] := new(Sieve)!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        assert isinstance(nodes[0], ClassDefNode)
        assign = nodes[1]
        assert isinstance(assign, IndexSetNode)
        assert assign.target == Identifier("Actor")
        assert assign.index == SymbolLiteralNode("Sam")
        assert isinstance(assign.value, CallNode)
        assert assign.value.func == Identifier("new")


class TestHistoricalActorFiles:
    """Parse real historical 1993 Whitewater Actor 4.1 files."""

    def test_parse_sieve_act(self):
        path = "historical_actor_41_source/actact/SIEVE.ACT"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) >= 4
        # Sieve defines ClassDefNode, NowNode, MethodDefNode, and Actor[#Sam] := new(Sieve)!!
        class_defs = [n for n in nodes if isinstance(n, ClassDefNode)]
        assert len(class_defs) == 1
        assert class_defs[0].name == "Sieve"
        assert class_defs[0].ivars == []

    def test_parse_sieve1_act(self):
        path = "historical_actor_41_source/actact/SIEVE1.ACT"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) >= 4
        class_defs = [n for n in nodes if isinstance(n, ClassDefNode)]
        assert len(class_defs) == 1
        assert class_defs[0].name == "Sieve"
        assert class_defs[0].ivars == []

    def test_parse_fractal_act(self):
        path = "historical_actor_41_source/actact/FRACTAL.ACT"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) >= 2

    def test_parse_button_cls(self):
        path = "historical_actor_41_source/actcls/BUTTON.CLS"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) == 12

    def test_parse_object_cls(self):
        path = "historical_actor_41_source/actcls/OBJECT.CLS"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) == 74

    def test_parse_fractalw_cls(self):
        path = "historical_actor_41_source/actcls/FRACTALW.CLS"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) == 6
        # Contains Action setLevels handler with range #[102-106]
        actions = [n for n in nodes if isinstance(n, MethodDefNode) and n.is_action]
        assert len(actions) == 1
        assert actions[0].method_name == "setLevels"
        assert ("range", 102, 106) in actions[0].action_selectors.elements


class TestMilestone1Remediation:
    """Regression tests for Milestone M1 Remediation:

    1. CARET return assignment precedence: `^var := expr;`
    2. Historical benchmarks and classes: LIST.ACT (17 nodes), ORDEREDC.CLS (27 nodes)
    3. DOS path string escaping and character literal backslash:
       - Single-quote backslash: '\\' (DOCFILE.CLS)
       - DOS directory paths ending with backslash: "res\\" (BROWSER.CLS)
       - Static C-string literals: %"..." and C-char %'...'
    4. Dual-mode discriminator comment & string stripping:
       - Nested block comments containing class Foo:
       - Multiline string literals containing class Foo:
       - Preserving Pythonic class compatibility for legacy code
    """

    def test_caret_return_assignment(self):
        source = """
        now(ListNode)!!
        Def setLeft(self, lft)
        {
            ^left := lft;
        }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        assert isinstance(nodes[0], NowNode)
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assert len(m.body) == 1
        ret = m.body[0]
        assert isinstance(ret, ReturnNode)
        assert isinstance(ret.value, AssignNode)
        assert ret.value.target == "left"
        assert ret.value.value == Identifier("lft")

    def test_caret_return_chained_and_indexed_assignment(self):
        source = """
        now(TestNode)!!
        Def testReturns(self, val)
        {
            ^self.field := val;
            ^arr[idx] := val;
            ^a := b := val;
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assert len(m.body) == 3
        assert isinstance(m.body[0], ReturnNode)
        assert isinstance(m.body[0].value, AssignNode)
        assert m.body[0].value.target == AttributeAccess(expr=SelfLiteral(), attribute="field")
        assert isinstance(m.body[1], ReturnNode)
        assert isinstance(m.body[1].value, IndexSetNode)
        assert isinstance(m.body[2], ReturnNode)
        assert isinstance(m.body[2].value, AssignNode)

    def test_parse_historical_list_act(self):
        path = "historical_actor_41_source/actact/LIST.ACT"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) == 17
        set_left_methods = [n for n in nodes if isinstance(n, MethodDefNode) and n.method_name == "setLeft"]
        assert len(set_left_methods) == 1
        assert isinstance(set_left_methods[0].body[0], ReturnNode)
        assert isinstance(set_left_methods[0].body[0].value, AssignNode)
        assert set_left_methods[0].body[0].value.target == "left"

    def test_parse_historical_orderedc_cls(self):
        path = "historical_actor_41_source/actcls/ORDEREDC.CLS"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) == 27
        set_last_methods = [n for n in nodes if isinstance(n, MethodDefNode) and n.method_name == "setLast"]
        assert len(set_last_methods) == 1
        assert isinstance(set_last_methods[0].body[0], ReturnNode)
        assert isinstance(set_last_methods[0].body[0].value, AssignNode)
        assert set_last_methods[0].body[0].value.target == "lastElement"

    def test_parse_historical_docfile_cls(self):
        path = "historical_actor_41_source/actcls/DOCFILE.CLS"
        if not os.path.exists(path):
            pytest.skip(f"Historical file {path} not found")
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        nodes = parse_actor(content)
        assert len(nodes) == 28

    def test_dos_path_and_backslash_character(self):
        source = """
        now(PathTest)!!
        Def testPaths(self | path, sep)
        {
            path := "res\\" + "file.wdl";
            sep := '\\';
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assigns = [s for s in m.body if isinstance(s, AssignNode)]
        assert len(assigns) == 2
        assert isinstance(assigns[0].value, BinaryOp)
        assert assigns[0].value.left == String("res\\")
        assert assigns[0].value.right == String("file.wdl")
        assert assigns[1].value == CharLiteral("\\")

    def test_browser_cls_dos_path_action(self):
        source = """
        now(Browser)!!
        Action editAttributes(self, msg | attr, editor) #[#browseAttr #ew->#ctrlT]
        {
            if (attr := attrName(selClass))
            then
                editor := loadTop(AttribEdit);
                loadStartFile(editor, "res\\" + attr + ".wdl");
            else
                ^nil;
            endif;
        }!!
        """
        nodes = parse_actor(source)
        assert len(nodes) == 2
        action_node = nodes[1]
        assert isinstance(action_node, MethodDefNode)
        assert action_node.is_action
        assert action_node.method_name == "editAttributes"

    def test_c_strings_and_c_chars(self):
        source = """
        now(CStringTest)!!
        Def testCString(self | s1, s2, c1, c2)
        {
            s1 := %"Hello\\r\\nWorld\\t\\"escaped\\"";
            s2 := %"\\251 1993";
            c1 := %'\\0';
            c2 := %'\\n';
        }!!
        """
        nodes = parse_actor(source)
        m = nodes[1]
        assert isinstance(m, MethodDefNode)
        assigns = [s for s in m.body if isinstance(s, AssignNode)]
        assert len(assigns) == 4
        assert assigns[0].value == String('Hello\r\nWorld\t"escaped"')
        assert assigns[1].value == String('\xa9 1993')
        assert assigns[2].value == CharLiteral('\0')
        assert assigns[3].value == CharLiteral('\n')

    def test_dual_mode_discriminator_nested_comments(self):
        source = """
        /* Outer comment
            /* Nested comment mentioning class Foo:
               which should not trigger Pythonic mode */
           back to outer */
        inherit(Object, #RealActor, nil, 2, nil)!!
        Def bar(self) { ^42; }!!
        """
        assert is_authentic_1993_actor(source) is True
        nodes = parse_actor(source)
        assert len(nodes) == 2
        assert isinstance(nodes[0], ClassDefNode)
        assert nodes[0].name == "RealActor"
        assert isinstance(nodes[1], MethodDefNode)

    def test_dual_mode_discriminator_multiline_string_fake_class(self):
        source = '''
        now(CodeGenerator)!!
        Def makeScript(self | code)
        {
            code := "line 1\\nclass FakePython:\\n    def method(): pass";
            ^code;
        }!!
        '''
        assert is_authentic_1993_actor(source) is True
        nodes = parse_actor(source)
        assert len(nodes) == 2
        assert isinstance(nodes[1], MethodDefNode)

    def test_dual_mode_preserves_pythonic_classes(self):
        source = """
        # Pythonic class for backward compatibility
        class LegacyActor:
            def __init__(self, val):
                self.val = val

            def get_val(self):
                return self.val
        """
        assert is_authentic_1993_actor(source) is False
        nodes = parse_actor(source)
        assert len(nodes) == 1
        assert nodes[0].name == "LegacyActor"

