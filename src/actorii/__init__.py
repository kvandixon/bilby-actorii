"""Main module for Project ActorII."""

def hello(name: str = "ActorII") -> str:
    """Return a greeting for the project."""
    return f"Hello, {name}! Welcome to ActorII."


if __name__ == "__main__":
    print(hello())