"""Basic tests for Project ActorII."""

from actorii.__init__ import hello


def test_hello():
    """Test the hello function."""
    result = hello()
    assert "ActorII" in result
    assert "Hello" in result


def test_hello_named():
    """Test the hello function with a name."""
    result = hello("TestUser")
    assert "TestUser" in result