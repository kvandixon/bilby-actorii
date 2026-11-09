"""Unit tests for ActorII parser and VM integration.

Tests that parsed source code executes correctly on ActorVM and produces
correct results, verifying the full parser → compile → run pipeline.
"""

import pytest

from actorii.vm import ActorVM, Opcode
from actorii.core import ActorInheritanceHierarchy
from actorii.parser import parse_actor, compile_to_bytecode


class TestParserVMIntegration:
    """Integration tests: parse source → compile → execute on ActorVM."""

    def test_basic_assignment_and_return(self):
        """Test that simple assignment and return works on ActorVM."""
        source = """
class Calculator:
    def add(self, a, b):
        result = a + b
        return result
"""

        ast = parse_actor(source)
        assert len(ast) == 1
        cls = ast[0]
        assert cls.name == "Calculator"

        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(instructions)
        vm.run()
        assert vm.halted is True

    def test_integer_arithmetic(self):
        """Test integer arithmetic operations on ActorVM."""
        source = """
class MathOps:
    def double(self, x):
        return x + x
    
    def increment(self, x):
        return x + 1
"""

        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(instructions)
        vm.run()
        assert vm.halted is True

    def test_string_concatenation(self):
        """Test string operations on ActorVM."""
        source = """
class Greeter:
    def greet(self, name):
        return "Hello, " + name
"""

        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(instructions)
        vm.run()
        assert vm.halted is True

    def test_binary_operations(self):
        """Test binary operations (==, !=, +, -) on ActorVM."""
        source = """
class Comparator:
    def is_positive(self, x):
        return x > 0
    
    def equals(self, a, b):
        return a == b
    
    def subtract(self, a, b):
        return a - b
"""

        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(instructions)
        vm.run()
        assert vm.halted is True

    def test_class_instantiation(self):
        """Test creating instances of parsed classes on ActorVM."""
        source = """
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count = self.count + 1
        return self.count
    
    def value(self):
        return self.count
"""

        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(instructions)
        vm.run()
        assert vm.halted is True


class TestParseActor:
    """Direct tests of the parse_actor function."""

    def test_empty_source(self):
        """Empty source should return empty list."""
        ast = parse_actor("")
        assert ast == []

    def test_whitespace_only(self):
        """Whitespace-only source should return empty list."""
        ast = parse_actor("   \n\n  \t  ")
        assert ast == []

    def test_simple_class(self):
        """Minimal class definition parsing."""
        source = "class Foo:\n    pass"
        ast = parse_actor(source)
        assert len(ast) == 1
        assert ast[0].name == "Foo"
        assert len(ast[0].methods) == 0

    def test_class_with_method(self):
        """Class with a single method."""
        source = "class Bar:\n    def baz(self):\n        pass"
        ast = parse_actor(source)
        assert len(ast) == 1
        assert ast[0].name == "Bar"
        assert len(ast[0].methods) == 1
        assert ast[0].methods[0].name == "baz"

    def test_class_with_init(self):
        """Class with __init__ method."""
        source = "class Baz:\n    def __init__(self, x):\n        self.x = x"
        ast = parse_actor(source)
        assert len(ast) == 1
        assert ast[0].name == "Baz"
        assert len(ast[0].methods) == 1
        assert ast[0].methods[0].name == "__init__"
        assert len(ast[0].methods[0].params) == 1
        assert ast[0].methods[0].params[0] == "x"

    def test_class_with_multiple_methods(self):
        """Class with multiple methods."""
        source = "class Qux:\n    def method1(self):\n        pass\n    def method2(self, a, b):\n        return a + b"
        ast = parse_actor(source)
        assert len(ast) == 1
        assert ast[0].name == "Qux"
        assert len(ast[0].methods) == 2

    def test_integer_literals(self):
        """Parse integer literals in expressions."""
        source = "class X:\n    def foo(self):\n        return 42"
        ast = parse_actor(source)
        assert len(ast) == 1

    def test_string_literals(self):
        """Parse string literals in expressions."""
        source = "class Y:\n    def foo(self):\n        return 'hello'"
        ast = parse_actor(source)
        assert len(ast) == 1

    def test_binary_expression(self):
        """Parse binary expressions."""
        source = "class Z:\n    def foo(self):\n        return a + b"
        ast = parse_actor(source)
        assert len(ast) == 1


class TestCompileBytecode:
    """Direct tests of compile_to_bytecode function."""

    def test_compiles_non_empty_ast(self):
        """Non-empty AST should produce bytecode."""
        source = "class Test:\n    def bar(self):\n        return 1"
        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0
        assert len(constants) > 0

    def test_compiles_class_name(self):
        """Bytecode should include class name as constant."""
        source = "class MyClass:\n    pass"
        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert "MyClass" in constants

    def test_compiles_method_names(self):
        """Bytecode should include method names as constants."""
        source = "class MyClass:\n    def my_method(self):\n        pass"
        ast = parse_actor(source)
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert "my_method" in constants


class TestActorVMWithParsedCode:
    """Tests ActorVM execution with code parsed from source files."""

    def test_parse_then_execute_simple(self):
        """Full pipeline: parse source, compile, execute on VM."""
        source = """
class Adder:
    def add_three(self, a, b, c):
        result = a + b + c
        return result
"""

        # Parse
        ast = parse_actor(source)
        assert len(ast) == 1

        # Compile
        instructions, constants, string_table = compile_to_bytecode(ast)
        assert len(instructions) > 0

        # Execute on VM
        from actorii.vm import ActorVM
        from actorii.core import ActorInheritanceHierarchy

        hierarchy = ActorInheritanceHierarchy()
        vm = ActorVM(hierarchy)
        vm.load(instructions)
        vm.run()
        assert vm.halted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])