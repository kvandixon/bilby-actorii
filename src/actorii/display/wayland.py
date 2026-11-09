# Wayland display backend for ActorII.
# Requires: pip install pywayland>=0.4.19
#
# NOTE: pywayland is a client library that connects to an existing Wayland
# compositor (Weston, GNOME, KDE, etc.). This module provides the API structure
# for Wayland buffer management, surface creation, and widget rendering.
# A real Wayland connection requires a running compositor.
#
# For testing/demo without a compositor, the module can operate in "headless"
# mode where buffer data is managed in memory without a real wl_display connection.

from __future__ import annotations


class WaylandDisplay:
    """Manages a Wayland display connection using pywayland.

    In headless/test mode, creates a simulated display connection for
    buffer and pixel operations without requiring a running compositor.
    """

    def __init__(self, headless: bool = False):
        self._headless = headless
        if headless:
            # Create a minimal surface for pixel operations without
            # requiring a real Wayland compositor
            self._display = None
            self._registry = None
            self._wl_shm = None
            self._xdg_wm_base = None
            print("   (headless mode: no real Wayland compositor needed)")
        else:
            # Real Wayland display — requires a running compositor
            # This will connect to the system Wayland display (e.g., :0)
            # and bind required global interfaces (wl_shm, xdg_wm_base, etc.)
            import warnings

            warnings.warn(
                "Real Wayland display requires a running compositor. "
                "Use headless=True for buffer/pixel operations demo."
            )
            self._display = None  # Would be WlDisplay()
            self._registry = None  # Would be display.get_registry()
            self._wl_shm = None  # Would bind wl_shm global
            self._xdg_wm_base = None  # Would bind xdg_wm_base global

    def create_surface(self, width: int, height: int) -> "WaylandSurface":
        """Create a new Wayland surface for rendering.

        Args:
            width: Surface width in pixels
            height: Surface height in pixels

        Returns:
            A WaylandSurface instance with a shared memory buffer.
        """
        if self._headless:
            # In headless mode, create a surface with an in-memory buffer
            # that simulates the wl_shm buffer without a real compositor
            buffer_data = {
                "width": width,
                "height": height,
                "stride": width * 4,
                "format": "xrgb8888",
                # Simulated buffer data — a bytearray large enough for
                # width*height pixels in xrgb8888 format (4 bytes per pixel)
                "data": bytearray(width * height * 4),
            }
            return WaylandSurface(
                wl_surface=None,
                xdg_surface=None,
                xdg_toplevel=None,
                buffer_data=buffer_data,
                headless=True,
            )
        else:
            # Real Wayland surface creation — would create wl_surface,
            # xdg_surface, xdg_toplevel, and wl_shm buffer via protocol
            # messages to the compositor
            raise NotImplementedError(
                "Real Wayland surface creation requires a running compositor. "
                "Use headless=True for demo."
            )


class WaylandSurface:
    """Wayland buffer surface for actor application rendering.

    Manages a wl_surface with shared memory buffers for pixel operations.
    Supports pixel-level get/set, damage tracking, and buffer presentation.
    In headless mode, uses an in-memory bytearray instead of real wl_shm.
    """

    def __init__(
        self,
        wl_surface,
        xdg_surface,
        xdg_toplevel,
        buffer_data: dict,
        headless: bool = False,
    ):
        self._wl_surface = wl_surface
        self._xdg_surface = xdg_surface
        self._xdg_toplevel = xdg_toplevel
        self._buffer = buffer_data
        self._headless = headless

        # Store surface dimensions from buffer data
        self._width = buffer_data["width"]
        self._height = buffer_data["height"]
        self._stride = buffer_data["stride"]
        self._format = buffer_data["format"]

        # In headless mode, the buffer data is a bytearray we can read/write
        if headless:
            self._buffer_data = buffer_data["data"]
            # Initialize all pixels to white (#FFFFFF = 0xFFFFFFFF in memory)
            # xrgb8888: byte 0 = X padding, bytes 1-3 = R G B
            # White = R=255, G=255, B=255, X=0
            white_pixel = bytes([0, 255, 255, 255])
            for i in range(0, len(self._buffer_data), 4):
                self._buffer_data[i : i + 4] = white_pixel

        # Commit initial state
        if not headless and self._wl_surface is not None:
            self._wl_surface.commit()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def stride(self):
        return self._stride

    @property
    def format(self):
        return self._format

    def get_pixel(self, x: int, y: int) -> str:
        """Get pixel color from the Wayland buffer.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Color as hex string (e.g., "#FF0000" for red)
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            return "#000000"

        # Calculate offset in shared memory:
        # xrgb8888 format: byte 0 = X padding, bytes 1-3 = R G B
        # stride is bytes per row, offset within row is x * 4
        offset = y * self._stride + x * 4

        if self._headless:
            # Read RGBA bytes from the in-memory buffer
            # buffer_data is a bytearray; pixel at (x,y) is at offset
            r = self._buffer_data[offset + 1]
            g = self._buffer_data[offset + 2]
            b = self._buffer_data[offset + 3]
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            # Read from real wl_shm buffer via pywayland
            # r = self._buffer["data"][offset + 1]
            # g = self._buffer["data"][offset + 2]
            # b = self._buffer["data"][offset + 3]
            raise NotImplementedError(
                "Reading from real Wayland buffer requires a running compositor"
            )

    def set_pixel(self, x: int, y: int, color: str) -> None:
        """Set pixel color in the Wayland buffer.

        Args:
            x: X coordinate
            y: Y coordinate
            color: Color as hex string (e.g., "#FF0000" for red)
        """
        if not (0 <= x < self._width and 0 <= y < self._height):
            return

        # Parse color
        if color.startswith("#"):
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
        else:
            r, g, b = 0, 0, 0

        # Write to shared memory at the correct offset
        # xrgb8888: byte 0 = X (padding), bytes 1-3 = R G B
        offset = y * self._stride + x * 4

        if self._headless:
            # Write RGBA bytes to in-memory buffer (byte 0 = padding/X, bytes 1-3 = R G B)
            self._buffer_data[offset] = 0  # X padding
            self._buffer_data[offset + 1] = r
            self._buffer_data[offset + 2] = g
            self._buffer_data[offset + 3] = b
        else:
            # Write to real wl_shm buffer via pywayland
            # self._buffer["data"][offset] = 0  # X padding
            # self._buffer["data"][offset + 1] = r
            # self._buffer["data"][offset + 2] = g
            # self._buffer["data"][offset + 3] = b
            raise NotImplementedError(
                "Writing to real Wayland buffer requires a running compositor"
            )

    def flush(self) -> None:
        """Present the Wayland buffer to the display compositor.

        Commits the surface, presenting the current buffer contents
        to the Wayland compositor for rendering.
        """
        if self._headless:
            # In headless mode, "flushing" just means the buffer data is
            # ready for the compositor to read. In a real implementation,
            # this would send the wl_surface.commit() protocol request.
            print("   (headless: buffer ready for compositor)")
        else:
            # Real Wayland: send commit protocol request to compositor
            # self._wl_surface.commit()
            pass

    def damage(
        self, x: int = 0, y: int = 0, w: int | None = None, h: int | None = None
    ) -> None:
        """Mark a region as needing redisplay on Wayland.

        Args:
            x: Left coordinate of damage region
            y: Top coordinate of damage region
            w: Width of damage region (defaults to surface width - x)
            h: Height of damage region (defaults to surface height - y)
        """
        if w is None:
            w = self._width - x
        if h is None:
            h = self._height - y

        if self._headless:
            print(f"   (headless: damage region at ({x},{y}), size {w}x{h})")
        else:
            # Real Wayland: send damage request to compositor
            # self._wl_surface.damage(x=x, y=y, width=w, height=h)
            pass

    def copy_from(
        self,
        source: "WaylandSurface",
        x: int = 0,
        y: int = 0,
        w: int | None = None,
        h: int | None = None,
        dst_x: int = 0,
        dst_y: int = 0,
    ) -> None:
        """Copy a region from another Wayland surface.

        Args:
            source: Source WaylandSurface to copy from
            x: Source left coordinate
            y: Source top coordinate
            w: Source region width (defaults to source.width - x)
            h: Source region height (defaults to source.height - y)
            dst_x: Destination left coordinate
            dst_y: Destination top coordinate
        """
        if w is None:
            w = min(source.width - x, self._width - dst_x)
        if h is None:
            h = min(source.height - y, self._height - dst_y)

        for dy in range(h):
            for dx in range(w):
                src_pixel = source.get_pixel(x + dx, y + dy)
                self.set_pixel(dst_x + dx, dst_y + dy, src_pixel)

    def clear(self, color: str = "#FFFFFF") -> None:
        """Clear the surface to the given color.

        Args:
            color: Color as hex string (default "#FFFFFF" / white)
        """
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        for py in range(self._height):
            for px in range(self._width):
                self.set_pixel(px, py, color)

    def post_event(self, event: dict) -> None:
        """Post an event to the Wayland surface.

        In headless mode, just records the event for tracking.
        In real Wayland mode, would send the appropriate protocol request
        to the compositor (e.g., key, pointer, axis events).

        Args:
            event: Event dict with type and relevant data.
                Expected keys: "type" ("keyboard", "mousebutton", "mousemove", etc.),
                    plus type-specific fields.
        """
        # In headless mode, we track events for demo purposes
        if self._headless:
            if not hasattr(self, "_posted_events"):
                self._posted_events = []
            self._posted_events.append(event)
            print(f"   (headless: posted event type={event.get('type', 'unknown')})")
        else:
            # Real Wayland: would send protocol requests to compositor
            # e.g., wl_keyboard.key, wl_pointer.axis, etc.
            pass

    def _create_buffer(self, width: int, height: int) -> dict:
        """Create a new shared memory buffer for the given dimensions.

        In headless mode, returns an in-memory bytearray buffer.
        In real Wayland mode, would create a wl_shm buffer via the compositor.

        Args:
            width: Buffer width in pixels
            height: Buffer height in pixels

        Returns:
            A buffer_data dict with width, height, stride, format, and data.
        """
        if self._headless:
            return {
                "width": width,
                "height": height,
                "stride": width * 4,
                "format": "xrgb8888",
                "data": bytearray(width * height * 4),
            }
        else:
            # Real Wayland: would create a wl_shm pool and buffer
            # via the protocol connection to the compositor
            raise NotImplementedError(
                "Real Wayland buffer creation requires a running compositor"
            )

    def resize(self, width: int, height: int) -> None:
        """Handle surface resize.

        Reallocates buffer bytearray, preserves existing pixel content
        where possible, initializes newly exposed pixels to white,
        and updates dimensions and stride.
        """
        w = max(1, int(width))
        h = max(1, int(height))
        old_w = self._width
        old_h = self._height
        old_stride = self._stride
        old_data = getattr(self, "_buffer_data", None)

        new_buffer = self._create_buffer(w, h)
        new_stride = new_buffer["stride"]
        new_data = new_buffer["data"]

        # Initialize all pixels to white (#FFFFFF in xrgb8888: 0, 255, 255, 255)
        white_pixel = bytes([0, 255, 255, 255])
        new_data[:] = white_pixel * (w * h)

        # Preserve overlapping pixel content
        if old_data is not None:
            copy_w = min(old_w, w)
            copy_h = min(old_h, h)
            row_bytes = copy_w * 4
            for y in range(copy_h):
                src_start = y * old_stride
                dst_start = y * new_stride
                new_data[dst_start : dst_start + row_bytes] = old_data[
                    src_start : src_start + row_bytes
                ]

        self._width = w
        self._height = h
        self._stride = new_stride
        self._buffer = new_buffer
        self._buffer_data = new_data

    @property
    def checked(self):
        """Checkbox state tracking (used by widget rendering)."""
        return getattr(self, "_checked", False)

    @checked.setter
    def checked(self, value: bool) -> None:
        self._checked = value


class WaylandDriver:
    """Concrete Wayland display driver for ActorII.

    Translates ActorII drawing operations to Wayland protocol commands.
    Maps input events and provides widget rendering support.
    """

    def __init__(self, surface: Surface):
        self._surface = surface
        self._wayland_surface = surface  # Reference to WaylandSurface

    @property
    def surface(self):
        """Get the underlying surface."""
        return self._wayland_surface

    def present(self) -> None:
        """Present the rendered buffer to the Wayland compositor."""
        if self._wayland_surface:
            self._wayland_surface.flush()
        else:
            self._surface.flush()

    def resize(self, width: int, height: int) -> None:
        """Handle surface resize via Wayland driver."""
        if hasattr(self._surface, "resize"):
            self._surface.resize(width, height)
        elif self._wayland_surface and hasattr(self._wayland_surface, "resize"):
            self._wayland_surface.resize(width, height)

    def handle_event(self, event: dict) -> bool:
        """Process Wayland input events.

        Maps:
        - key events → ActorII keyboard messages
        - pointer events → ActorII mouse messages

        Returns:
            True if the event was handled, False otherwise.
        """
        event_type = event.get("type", "")

        if event_type == "key":
            key = event.get("key", "")
            # Post keyboard event to actor language surface
            self._wayland_surface.post_event(
                {
                    "type": "keyboard",
                    "key": key,
                    "down": event.get("down", True),
                }
            )
            return True

        elif event_type == "pointer":
            action = event.get("action", "")
            x = event.get("x", 0)
            y = event.get("y", 0)

            if action == "press":
                self._wayland_surface.post_event(
                    {
                        "type": "mousebutton",
                        "pressed": True,
                        "pos": (x, y),
                    }
                )
            elif action == "release":
                self._wayland_surface.post_event(
                    {
                        "type": "mousebutton",
                        "pressed": False,
                        "pos": (x, y),
                    }
                )
            elif action == "move":
                self._wayland_surface.post_event(
                    {
                        "type": "mousemove",
                        "pos": (x, y),
                    }
                )
            return True

        return False

    def clear(self, color: str = "#FFFFFF") -> None:
        """Clear the Wayland surface to the given color."""
        if self._wayland_surface:
            self._wayland_surface.clear(color)

    def register_object(self, obj) -> None:
        """Register an actor language object on this surface."""
        if not hasattr(self, "_registered_objects"):
            self._registered_objects = set()
        self._registered_objects.add(id(obj))

    def unregister_object(self, obj) -> None:
        """Unregister an actor language object."""
        if hasattr(self, "_registered_objects"):
            self._registered_objects.discard(id(obj))


# Factory function
def create_driver(mode: str = "auto", **kwargs):
    """Create a display driver for the given mode.

    Args:
        mode: "wayland", "websocket", or "auto"
        **kwargs: Passed to the concrete driver constructor
            - width: Surface width in pixels (default: 800)
            - height: Surface height in pixels (default: 600)
            - headless: If True, creates a headless display (no compositor needed)

    Returns:
        A DisplayDriver instance for the requested backend.

    Example:
        >>> driver = create_driver("wayland", width=800, height=600, headless=True)
        >>> surface = driver.surface
        >>> surface.set_pixel(100, 100, "#FF0000")
        >>> driver.present()
    """
    headless = kwargs.pop("headless", False)

    if mode == "wayland":
        from actorii.display.wayland import (
            WaylandDisplay,
            WaylandSurface,
            WaylandDriver,
        )

        display = WaylandDisplay(headless=headless)
        ws = display.create_surface(kwargs.get("width", 800), kwargs.get("height", 600))
        return WaylandDriver(ws)
    elif mode == "websocket":
        from actorii.display.websocket import WebSocketDriver

        return WebSocketDriver(kwargs.get("surface"))
    else:  # auto
        try:
            from actorii.display.wayland import (
                WaylandDisplay,
                WaylandSurface,
                WaylandDriver,
            )

            display = WaylandDisplay(headless=headless)
            ws = display.create_surface(
                kwargs.get("width", 800), kwargs.get("height", 600)
            )
            return WaylandDriver(ws)
        except Exception:
            from actorii.display.websocket import WebSocketDriver

            return WebSocketDriver(kwargs.get("surface"))
