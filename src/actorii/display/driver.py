# Abstract base classes for the ActorII display driver layer.

from abc import ABC, abstractmethod
from actorii.core import ActorObject


class DisplayDriver(ABC):
    """Abstract base for all display drivers.

    The 1993 actor language was Windows-native with GDI-based rendering.
    This abstraction adapts those concepts to modern display backends.
    """

    def __init__(self, surface):
        self._surface = surface

    @property
    @abstractmethod
    def surface(self):
        """The render surface managed by this driver."""
        pass

    @abstractmethod
    def present(self):
        """Present the rendered surface buffer to the display hardware.

        Corresponds to 1993 GDI's BitBlt/StretchBlt present operation.
        Wayland: wl_surface.commit()
        WebSocket: send 'flush' message
        """
        pass

    @abstractmethod
    def resize(self, width, height):
        """Handle surface resize events.

        The 1993 actor language responded to WM_SIZE messages.
        Modern backends must adjust buffer geometry and coordinate spaces.
        """
        pass

    @abstractmethod
    def handle_event(self, event):
        """Process input events from the display system.

        1993 event types mapped:
        - WM_KEYDOWN/KEYUP -> keyboard events
        - WM_LBUTTONDOWN/UP/DBLCLK -> mouse button events
        - WM_MOUSEMOVE -> mouse motion
        - WM_PAINT -> redraw request

        Returns True if event was handled, False for propagation.
        """
        pass

    @abstractmethod
    def clear(self, color="#FFFFFF"):
        """Clear the surface to a background color.

        Corresponds to 1993 GDI's PatBlt with BRUSH/SOLID pattern.
        """
        pass

    def register_object(self, obj):
        """Register an actor language object on this surface.

        Objects can query the surface for rendering purposes.
        Default no-op; concrete drivers may optimize.
        """
        self._surface.register_object(obj)

    def unregister_object(self, obj):
        """Unregister an actor language object."""
        self._surface.unregister_object(obj)


class Surface(ABC):
    """Abstract display surface representing a render target.

    The 1993 actor language had window surfaces (WIN*.CLS) and
    screen buffers. This abstraction adapts those concepts.
    """

    @property
    @abstractmethod
    def width(self):
        """Surface width in pixels."""
        pass

    @property
    @abstractmethod
    def height(self):
        """Surface height in pixels."""
        pass

    @property
    @abstractmethod
    def stride(self):
        """Bytes per row (stride/pitch). May be wider than width*4
        for alignment purposes."""
        pass

    @property
    @abstractmethod
    def format(self):
        """Pixel format string (e.g., 'argb32', 'rgb24', 'xrgb8888')."""
        pass

    @abstractmethod
    def get_pixel(self, x, y):
        """Get pixel color at coordinates (x, y).

        Returns color as hex string (e.g., '#RRGGBB' or '#AARRGGBB').
        """
        pass

    @abstractmethod
    def set_pixel(self, x, y, color):
        """Set pixel color at coordinates (x, y)."""
        pass

    @abstractmethod
    def flush(self):
        """Present the completed frame to the display.

        Wayland: commit the buffer
        WebSocket: send 'flush' command
        """
        pass

    @abstractmethod
    def damage(self, x=0, y=0, w=None, h=None):
        """Mark a region as needing redisplay.

        Wayland: wl_surface.damage(x, y, w, h)
        WebSocket: send 'damage' message with region
        """
        pass

    @abstractmethod
    def copy_from(self, source, x=0, y=0, w=None, h=None, dst_x=0, dst_y=0):
        """Copy a region from another surface.

        1993 equivalent: BitBlt/SrcCopy.
        """
        pass


class GraphicsContext(ABC):
    """Abstract graphics context for drawing operations.

    Corresponds to 1993 GDC (Graphics Device Context) patterns from
    ACTMAIN.$ and window class definitions.
    """

    def __init__(self, surface):
        self._surface = surface

    @property
    @abstractmethod
    def surface(self):
        """The surface this context draws on."""
        pass

    @abstractmethod
    def move_to(self, x, y):
        """Move the current pen position without drawing."""
        pass

    @abstractmethod
    def line_to(self, x, y):
        """Draw a line from current pen position to (x, y)."""
        pass

    @abstractmethod
    def fill_rect(self, x, y, w, h, color):
        """Fill a rectangle at (x, y) with width w, height h in the given color."""
        pass

    @abstractmethod
    def fill_poly(self, points, color):
        """Fill a polygon defined by vertex points."""
        pass

    @abstractmethod
    def set_color(self, color):
        """Set the current drawing color for subsequent operations."""
        pass

    @abstractmethod
    def set_font(self, font_spec):
        """Set the font for subsequent text operations.

        1993 equivalent: SelectObject(hFont) into GDC.
        Wayland: Pango font description or Cairo font face
        Web: Canvas 2D font or pre-rendered glyph texture
        """
        pass

    @abstractmethod
    def text_out(self, x, y, string):
        """Draw a text string at position (x, y)."""
        pass

    @abstractmethod
    def fill_ellipse(self, x, y, w, h, color):
        """Fill an ellipse bounding box at (x, y) with width w, height h."""
        pass