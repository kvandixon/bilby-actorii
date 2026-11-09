import pytest
from actorii.core import ActorInheritanceHierarchy
from actorii.vm import ActorVM, Opcode

def test_vm_push_pop_basic():
    hierarchy = ActorInheritanceHierarchy()
    vm = ActorVM(hierarchy)
    instructions = [
        (Opcode.PUSH_INT, 42),
        (Opcode.PUSH_INT, 10),
        (Opcode.POP, 0),
        (Opcode.RETURN, 0),
    ]
    vm.load(instructions)
    vm.run()
    assert vm.halted is True
    assert len(vm.stack) == 1
    assert vm.stack[0] == 42

def test_vm_object_lifecycle():
    hierarchy = ActorInheritanceHierarchy()
    vm = ActorVM(hierarchy)
    instructions = [
        (Opcode.NEW_OBJ, 0),
        (Opcode.RETURN, 0),
    ]
    vm.load(instructions)
    vm.run()
    assert vm.halted is True
    assert len(vm.objects) == 1
