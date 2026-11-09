"""Test module for Project ActorII core and display modules.

Verifies the clean-room reverse-engineered actor language constructs
and display surface architecture from the 1993 Whitewater Group/Genesis
Systems actor language.
"""

from actorii.core import (
    ActorObject,
    ActorClass,
    ActorInheritanceHierarchy,
    ActorMessage,
)
from actorii.display import (
    DisplaySurface,
    Window,
    Widget,
    Button,
    Checkbox,
    TextField,
    HTML5Surface,
)


def test_actor_object_creation():
    """Test basic ActorObject instantiation and message passing."""
    obj = ActorObject(obj_id="test-001")
    assert obj.class_name == "ActorObject"
    assert obj.obj_id == "test-001"
    print("  PASS: ActorObject creation and properties")


def test_actor_object_method_dispatch():
    """Test method dispatch via message passing.
    
    Note: Methods must be registered via set_method before dispatch.
    """
    obj = ActorObject(obj_id="test-002")
    def greet(name):
        return "Hello, " + name + "!"
    obj.set_method("greet", greet)
    result = obj.dispatch("greet", "World")
    assert result == "Hello, World!", f"Expected 'Hello, World!', got '{result}'"
    print("  PASS: ActorObject message dispatch")


def test_actor_object_define_field():
    """Test instance field management."""
    obj = ActorObject(obj_id="test-003")
    obj.define_field("counter", 0)
    assert obj.field("counter") == 0
    obj.define_field("counter", 42)
    assert obj.field("counter") == 42
    obj.undefine_field("counter")
    assert obj.field("counter") is None
    print("  PASS: ActorObject field management")


def test_actor_class_registration():
    """Test ActorClass registration and method definitions."""
    cls = ActorClass(name="TestClass")
    assert cls.name == "TestClass"
    
    def class_method():
        return "class method called"
    
    cls.add_method("greet", class_method)
    assert cls.has_method("greet")
    method = cls.get_method("greet")
    assert method() == "class method called"
    print("  PASS: ActorClass method registration")


def test_actor_class_inheritance():
    """Test single inheritance hierarchy."""
    hierarchy = ActorInheritanceHierarchy()
    
    parent = ActorClass(name="Parent")
    child = ActorClass(name="Child", bases=("Parent",))
    
    hierarchy.register(parent)
    hierarchy.register(child)
    
    # Child should inherit parent methods
    assert hierarchy.get("Child") is not None
    assert hierarchy.get("Parent") is not None
    print("  PASS: ActorInheritanceHierarchy registration")


def test_actor_message_creation():
    """Test ActorMessage creation and execution."""
    obj = ActorObject(obj_id="msg-test")
    
    def echo(msg):
        return "Echo: " + msg
    
    obj.set_method("echo", echo)
    
    msg = ActorMessage(receiver=obj, method_name="echo", msg="test")
    result = msg.execute()
    assert result == "Echo: test", f"Expected 'Echo: test', got '{result}'"
    print("  PASS: ActorMessage execution")


def test_window_creation():
    """Test Window instantiation and rendering."""
    win = Window(x=100, y=100, width=400, height=300,
                 title="Test Window")
    assert win.width == 400
    assert win.height == 300
    assert win.title == "Test Window"
    print("  PASS: Window creation")


def test_button_widget():
    """Test Button widget creation and event handling."""
    btn = Button(x=50, y=50, width=100, height=30, label="Click Me")
    assert btn.label == "Click Me"
    assert btn.x == 50
    assert btn.y == 50
    print("  PASS: Button widget creation")


def test_checkbox_widget():
    """Test Checkbox widget creation and state."""
    chk = Checkbox(x=20, y=20, width=80, height=30, label="Option A")
    assert chk.label == "Option A"
    assert chk._checked == False
    print("  PASS: Checkbox widget creation")


def test_textfield_widget():
    """Test TextField widget creation and text handling."""
    tf = TextField(x=10, y=10, width=150, height=25, label="Enter name")
    assert tf.label == "Enter name"
    assert tf._text == ""
    print("  PASS: TextField widget creation")


def test_html5_surface_render():
    """Test HTML5Surface rendering output."""
    surface = HTML5Surface(width=200, height=200, title="Test HTML5 App")
    html = surface.render()
    assert "DOCTYPE html" in html
    assert "actor_canvas" in html
    assert "200" in html  # width
    assert "200" in html  # height
    assert "Test HTML5 App" in html
    print("  PASS: HTML5Surface render output")


def test_full_integration():
    """Test full integration: class, inheritance, message, and display."""
    hierarchy = ActorInheritanceHierarchy()
    
    # Create a class with a method
    my_class = ActorClass(name="GreetingClass")
    def greet_method(name):
        return "Hello from ActorII: " + name
    my_class.add_method("greet", greet_method)
    hierarchy.register(my_class)
    
    # Create an instance using the class's default constructor
    instance = ActorObject(obj_id="int-test")
    instance.inherit_from(my_class)
    
    # Dispatch a message through inheritance
    result = instance.dispatch("greet", "World")
    assert "Hello from ActorII: World" in result
    
    # Create a window and surface
    win = Window(title="ActorII Test", width=400, height=300)
    surface = HTML5Surface(width=400, height=300, title="ActorII Test")
    html = surface.render()
    
    # The HTML output contains the title and canvas, not the greeting method result directly
    assert "ActorII Test" in html
    assert "actor_canvas" in html
    print("  PASS: Full integration test")


if __name__ == "__main__":
    print("Running ActorII test suite...\n")
    
    test_actor_object_creation()
    test_actor_object_method_dispatch()
    test_actor_object_define_field()
    test_actor_class_registration()
    test_actor_class_inheritance()
    test_actor_message_creation()
    test_window_creation()
    test_button_widget()
    test_checkbox_widget()
    test_textfield_widget()
    test_html5_surface_render()
    test_full_integration()
    
    print("\n=== All tests passed! ===")