"""Entry point for Project ActorII command-line usage."""

from actorii.__init__ import hello


def main() -> None:
    """Run the ActorII main routine."""
    print(hello("ActorII"))


if __name__ == "__main__":
    main()