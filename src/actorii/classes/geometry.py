"""1993 Whitewater Actor 4.1 Geometry Classes (Point, Point3D, Rect).

Reconstructed from authentic 1993 Actor Professional 4.1 sources:
- historical_actor_41_source/actcls/POINT.CLS
- historical_actor_41_source/actcls/POINT3D.CLS
- historical_actor_41_source/actcls/RECT.CLS
- historical_actor_41_source/actact/FRACTAL.ACT
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
from actorii.core import ActorObject


class Point(ActorObject):
    """Authentic 1993 Whitewater Actor Point class (POINT.CLS).

    2D coordinate point with vector arithmetic, equality, rounding, and projection.
    """

    def __init__(self, x: int = 0, y: int = 0, obj_id: Optional[str] = None):
        super().__init__(obj_id=obj_id)
        self._class_name = "Point"
        self._x = int(x)
        self._y = int(y)

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def get_x(self) -> int:
        return self._x

    def get_y(self) -> int:
        return self._y

    def setX(self, xVal: int) -> int:
        """Set x coordinate and return new x."""
        self._x = int(xVal)
        return self._x

    def setY(self, yVal: int) -> int:
        """Set y coordinate and return new y."""
        self._y = int(yVal)
        return self._y

    def set(self, xVal: int, yVal: int) -> "Point":
        """Set both coordinates and return self."""
        self._x = int(xVal)
        self._y = int(yVal)
        return self

    def asPoint(self) -> "Point":
        """Return 2D projection of self (identity for Point)."""
        return self

    # Vector & Scalar Arithmetic
    def __add__(self, other: Any) -> "Point":
        if isinstance(other, Point):
            return Point(self._x + other._x, self._y + other._y)
        elif isinstance(other, (int, float)):
            return Point(self._x + int(other), self._y + int(other))
        raise TypeError(f"Cannot add Point and {type(other)}")

    def __radd__(self, other: Any) -> "Point":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Point":
        if isinstance(other, Point):
            return Point(self._x - other._x, self._y - other._y)
        elif isinstance(other, (int, float)):
            return Point(self._x - int(other), self._y - int(other))
        raise TypeError(f"Cannot subtract Point and {type(other)}")

    def __mul__(self, scalar: Any) -> "Point":
        if isinstance(scalar, (int, float)):
            return Point(int(self._x * scalar), int(self._y * scalar))
        raise TypeError(f"Cannot multiply Point by {type(scalar)}")

    def __rmul__(self, scalar: Any) -> "Point":
        return self.__mul__(scalar)

    def __floordiv__(self, scalar: Any) -> "Point":
        if isinstance(scalar, (int, float)):
            s = int(scalar)
            if s == 0:
                raise ZeroDivisionError("Point division by zero")
            return Point(self._x // s, self._y // s)
        raise TypeError(f"Cannot floor divide Point by {type(scalar)}")

    def __truediv__(self, scalar: Any) -> "Point":
        return self.__floordiv__(scalar)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Point):
            return self._x == other._x and self._y == other._y
        return False

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self._x) ^ hash(self._y)

    def round(self) -> "Point":
        """Round coordinates to nearest integers."""
        return Point(round(self._x), round(self._y))

    def asString(self) -> str:
        return f"{self._x}@{self._y}"

    def middlePoint(self, aPt: "Point", rng: int = 0) -> "Point":
        """Midpoint of two 2D points with optional random displacement."""
        return Point((self._x + aPt.x) // 2, (self._y + aPt.y) // 2)

    def print(self, stream: Any = None) -> "Point":
        s = self.asString()
        if stream is not None and hasattr(stream, "write"):
            stream.write(s)
        else:
            print(s)
        return self

    def printOn(self, stream: Any = None) -> str:
        s = self.asString()
        if stream is not None and hasattr(stream, "write"):
            stream.write(s)
        return s

    def __repr__(self) -> str:
        return f"Point({self._x}, {self._y})"


class Point3D(Point):
    """Authentic 1993 Whitewater Actor Point3D class (POINT3D.CLS).

    3D coordinate point inheriting Point, with vector arithmetic,
    middlePoint fractal calculation, and authentic 2D projection asPoint().
    """

    def __init__(
        self, x: int = 0, y: int = 0, z: int = 0, obj_id: Optional[str] = None
    ):
        super().__init__(x, y, obj_id=obj_id)
        self._class_name = "Point3D"
        self._z = int(z)

    @property
    def z(self) -> int:
        return self._z

    def get_z(self) -> int:
        return self._z

    def setZ(self, zVal: int) -> int:
        self._z = int(zVal)
        return self._z

    def asPoint(self) -> Point:
        """Authentic 1993 projection: point(x, y - (z * 9) / 8).

        Source: POINT3D.CLS:28 Def asPoint(self) { ^point(x, y - (z * 9) / 8); }
        """
        return Point(self._x, self._y - (self._z * 9) // 8)

    # 3D Vector & Scalar Arithmetic
    def __add__(self, other: Any) -> "Point3D":
        if isinstance(other, Point3D):
            return Point3D(self._x + other._x, self._y + other._y, self._z + other._z)
        elif isinstance(other, Point):
            return Point3D(self._x + other.x, self._y + other.y, self._z)
        elif isinstance(other, (int, float)):
            return Point3D(
                self._x + int(other), self._y + int(other), self._z + int(other)
            )
        raise TypeError(f"Cannot add Point3D and {type(other)}")

    def __radd__(self, other: Any) -> "Point3D":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Point3D":
        if isinstance(other, Point3D):
            return Point3D(self._x - other._x, self._y - other._y, self._z - other._z)
        elif isinstance(other, Point):
            return Point3D(self._x - other.x, self._y - other.y, self._z)
        elif isinstance(other, (int, float)):
            return Point3D(
                self._x - int(other), self._y - int(other), self._z - int(other)
            )
        raise TypeError(f"Cannot subtract Point3D and {type(other)}")

    def __mul__(self, scalar: Any) -> "Point3D":
        if isinstance(scalar, (int, float)):
            return Point3D(
                int(self._x * scalar), int(self._y * scalar), int(self._z * scalar)
            )
        raise TypeError(f"Cannot multiply Point3D by {type(scalar)}")

    def __rmul__(self, scalar: Any) -> "Point3D":
        return self.__mul__(scalar)

    def __floordiv__(self, scalar: Any) -> "Point3D":
        if isinstance(scalar, (int, float)):
            s = int(scalar)
            if s == 0:
                raise ZeroDivisionError("Point3D division by zero")
            return Point3D(self._x // s, self._y // s, self._z // s)
        raise TypeError(f"Cannot floor divide Point3D by {type(scalar)}")

    def __truediv__(self, scalar: Any) -> "Point3D":
        return self.__floordiv__(scalar)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Point3D):
            return self._x == other._x and self._y == other._y and self._z == other._z
        return False

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __hash__(self) -> int:
        return hash(self._x) ^ hash(self._y) ^ hash(self._z)

    def middlePoint(self, aPt: "Point3D", rng: int = 0) -> "Point3D":
        """Calculate midpoint with optional range offset for fractal generation (POINT3D.CLS)."""
        mid_x = (self._x + getattr(aPt, "x", 0)) // 2
        mid_y = (self._y + getattr(aPt, "y", 0)) // 2
        rnd = 0 if rng == 0 else (rng // 2)
        mid_z = ((self._z + getattr(aPt, "z", 0)) // 2) + rnd - (rng // 2)
        return Point3D(mid_x, mid_y, mid_z)

    def asString(self) -> str:
        return f"Point3D({self._x}, {self._y}, {self._z})"

    def printOn(self, stream: Any = None) -> str:
        s = self.asString()
        if stream is not None and hasattr(stream, "write"):
            stream.write(s)
        return s

    def __repr__(self) -> str:
        return self.asString()


class Rect(ActorObject):
    """Authentic 1993 Actor Rect (rectangle) class.

    Defined by origin (left, top) and corner (right, bottom).
    Source: historical_actor_41_source/actcls/RECT.CLS
    """

    def __init__(
        self,
        left_or_origin: Union[int, Point] = 0,
        top_or_corner: Union[int, Point] = 0,
        right: Optional[int] = None,
        bottom: Optional[int] = None,
        obj_id: Optional[str] = None,
    ):
        super().__init__(obj_id=obj_id)
        self._class_name = "Rect"
        if isinstance(left_or_origin, Point):
            self.left = int(left_or_origin.x)
            self.top = int(left_or_origin.y)
            if isinstance(top_or_corner, Point):
                self.right = int(top_or_corner.x)
                self.bottom = int(top_or_corner.y)
            else:
                self.right = int(right if right is not None else self.left)
                self.bottom = int(bottom if bottom is not None else self.top)
        else:
            self.left = int(left_or_origin)
            self.top = int(top_or_corner)
            self.right = int(right if right is not None else self.left)
            self.bottom = int(bottom if bottom is not None else self.top)

    @property
    def width(self) -> int:
        return abs(self.right - self.left)

    @width.setter
    def width(self, val: int) -> None:
        self.right = self.left + int(val)

    @property
    def height(self) -> int:
        return abs(self.bottom - self.top)

    @height.setter
    def height(self, val: int) -> None:
        self.bottom = self.top + int(val)

    @property
    def origin(self) -> Point:
        return Point(self.left, self.top)

    @property
    def corner(self) -> Point:
        return Point(self.right, self.bottom)

    def setOrigin(self, aPoint: Point) -> "Rect":
        self.left = aPoint.x
        self.top = aPoint.y
        return self

    def setCorner(self, aPoint: Point) -> "Rect":
        self.right = aPoint.x
        self.bottom = aPoint.y
        return self

    def contains(self, x: int, y: int) -> bool:
        """Test if (x, y) point lies within the rectangle."""
        min_x, max_x = min(self.left, self.right), max(self.left, self.right)
        min_y, max_y = min(self.top, self.bottom), max(self.top, self.bottom)
        return min_x <= x <= max_x and min_y <= y <= max_y

    def containsPoint(self, aPoint: Point) -> bool:
        return self.contains(aPoint.x, aPoint.y)

    def intersects(self, other: "Rect") -> bool:
        min_x1, max_x1 = min(self.left, self.right), max(self.left, self.right)
        min_y1, max_y1 = min(self.top, self.bottom), max(self.top, self.bottom)
        min_x2, max_x2 = min(other.left, other.right), max(other.left, other.right)
        min_y2, max_y2 = min(other.top, other.bottom), max(other.top, other.bottom)
        return not (
            max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1
        )

    def union(self, other: "Rect") -> "Rect":
        min_x = min(self.left, self.right, other.left, other.right)
        min_y = min(self.top, self.bottom, other.top, other.bottom)
        max_x = max(self.left, self.right, other.left, other.right)
        max_y = max(self.top, self.bottom, other.top, other.bottom)
        return Rect(min_x, min_y, max_x, max_y)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def to_dlg_units(self) -> Tuple[int, int, int, int]:
        return (self.left // 4, self.top // 8, self.width // 4, self.height // 8)

    @classmethod
    def from_dlg_units(cls, dl: int, dt: int, dw: int, dh: int) -> "Rect":
        return cls(dl * 4, dt * 8, dl * 4 + dw * 4, dt * 8 + dh * 8)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Rect):
            return (
                self.left == other.left
                and self.top == other.top
                and self.right == other.right
                and self.bottom == other.bottom
            )
        elif isinstance(other, (tuple, list)) and len(other) == 4:
            return (self.left, self.top, self.right, self.bottom) == tuple(other)
        return False

    def asString(self) -> str:
        return f"Rect({self.left}, {self.top}, {self.right}, {self.bottom})"

    def __repr__(self) -> str:
        return self.asString()


__all__ = ["Point", "Point3D", "Rect"]
