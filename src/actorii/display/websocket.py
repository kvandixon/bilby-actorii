# WebSocket-mediated canvas rendering backend for ActorII.

# The actor application runs in a browser, with rendering commands
# transmitted via WebSocket from the Python runtime to a JavaScript
# canvas runtime. This enables HTML5 deployment without requiring
# a native display server.

from __future__ import annotations
import json
import asyncio
from actorii.display.driver import DisplayDriver, Surface, GraphicsContext


class WebSocketSurface(Surface):
    """HTML5 canvas surface driven via WebSocket commands.

    The Python actor runtime sends rendering commands through a WebSocket
    connection to a JavaScript client that draws on an HTML5 canvas.
    """

    def __init__(self, width: int = 800, height: int = 600, title: str = "Actor Application"):
        self._width = width
        self._height = height
        self._title = title
        self._canvas_id = f"actor_canvas_{id(self)}"
        self._commands = []  # Accumulated render commands
        self._dirty = True  # Whether the frame needs presenting
        self._pixel_data = [["#FFFFFF"] * width for _ in range(height)]

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def stride(self):
        return self._width * 4  # 4 bytes per pixel (argb32)

    @property
    def format(self):
        return "argb32"

    def get_pixel(self, x, y):
        if 0 <= x < self._width and 0 <= y < self._height:
            return self._pixel_data[y][x]
        return "#000000"

    def set_pixel(self, x, y, color):
        if 0 <= x < self._width and 0 <= y < self._height:
            self._pixel_data[y][x] = color
            self._commands.append({"type": "pixel", "x": x, "y": y, "color": color})
            self._dirty = True

    def flush(self):
        """Mark the frame as presented.

        In the WebSocket protocol, this sends a 'flush' command
        to the JavaScript runtime, which presents the completed frame.
        """
        self._dirty = True
        # Will be sent via WebSocket by the driver

    def damage(self, x=0, y=0, w=None, h=None):
        """Mark a region as needing redisplay.

        Sends a 'damage' WebSocket command with the region coordinates.
        """
        if w is None:
            w = self._width - x
        if h is None:
            h = self._height - y
        self._commands.append({
            "type": "damage",
            "x": x,
            "y": y,
            "w": w,
            "h": h,
        })
        self._dirty = True

    def copy_from(self, source, x=0, y=0, w=None, h=None, dst_x=0, dst_y=0):
        """Copy a region from another surface.

        1993 equivalent: BitBlt/SrcCopy.
        Accumulates a 'copy' command for WebSocket transmission.
        """
        if w is None:
            w = min(self._width - x, source.width - dst_x)
        if h is None:
            h = min(self._height - y, source.height - dst_y)
        self._commands.append({
            "type": "copy",
            "src": source,
            "src_x": x,
            "src_y": y,
            "dst_x": dst_x,
            "dst_y": dst_y,
            "w": w,
            "h": h,
        })
        self._dirty = True

    def _serialize_commands(self):
        """Convert accumulated commands to WebSocket JSON payload."""
        # Group consecutive same-type commands for efficiency
        if not self._commands:
            return json.dumps({"type": "noop"})

        # For a production implementation, we'd batch and compress
        # But for now, just return the command list
        return json.dumps(self._commands)


class WebSocketDriver(DisplayDriver):
    """Concrete WebSocket display driver for ActorII.

    Translates ActorII drawing operations to WebSocket messages
    sent to a browser-based JavaScript canvas runtime.
    """

    def __init__(self, surface: WebSocketSurface, ws_connection=None):
        super().__init__(surface)
        self._surface = surface
        self._ws = ws_connection  # Will be a websocket.WebSocketConnection
        self._command_buffer = []

    def present(self):
        """Send flush command via WebSocket to present the frame."""
        if self._ws and self._ws.open:
            cmd = self._surface._serialize_commands()
            # Wrap in a flush message
            msg = json.dumps({"type": "flush", "commands": cmd})
            self._ws.send(msg)
        else:
            # If no WS connection, just mark dirty locally
            self._surface._dirty = True

    def resize(self, width, height):
        """Handle surface resize via WebSocket."""
        self._surface._width = width
        self._surface._height = height
        self._surface._pixel_data = [
            ["#FFFFFF"] * width for _ in range(height)
        ]
        self._surface._commands.append({
            "type": "resize",
            "width": width,
            "height": height,
        })
        self._surface._dirty = True

    def handle_event(self, event):
        """Process input events from the WebSocket/client.

        Received message types:
        - 'event': {type, x, y, key, state} from browser JavaScript
        - 'resize': {width, height} from browser window resize

        Maps to ActorII event types:
        - keyboard → key down/up messages
        - mousebutton → click/press/release messages
        - mousemove → mouse motion messages
        """
        msg_type = event.get("type", "") if isinstance(event, dict) else ""

        if msg_type == "event":
            event_data = event.get("data", {})
            etype = event_data.get("type", "")

            if etype == "keyboard":
                key = event_data.get("key", "")
                down = event_data.get("down", True)
                self._surface.post_event({
                    "type": "keyboard",
                    "key": key,
                    "down": down,
                })
                return True

            elif etype == "mousebutton":
                pressed = event_data.get("pressed", False)
                x = event_data.get("x", 0)
                y = event_data.get("y", 0)
                self._surface.post_event({
                    "type": "mousebutton",
                    "pressed": pressed,
                    "pos": (x, y),
                })
                return True

            elif etype == "mousemove":
                x = event_data.get("x", 0)
                y = event_data.get("y", 0)
                self._surface.post_event({
                    "type": "mousemove",
                    "pos": (x, y),
                })
                return True

        elif msg_type == "resize":
            width = event.get("width", self._surface.width)
            height = event.get("height", self._surface.height)
            self.resize(width, height)
            return True

        return False

    def clear(self, color="#FFFFFF"):
        """Clear the WebSocket surface to the given color."""
        self._surface.set_pixel(0, 0, color)  # Will clear all pixels via accumulated commands
        # Actually, send a clear command
        if self._ws and self._ws.open:
            msg = json.dumps({"type": "clear", "color": color})
            self._ws.send(msg)
        self._surface._pixel_data = [
            ["#FFFFFF"] * self._surface.width for _ in range(self._surface.height)
        ]

    def send_commands(self, commands):
        """Batch send multiple render commands via WebSocket.

        Called by the actor language VM after executing drawing operations.
        Commands should be in the WebSocket protocol format.
        """
        if not self._ws or not self._ws.open:
            return

        # Batch commands into a single message for efficiency
        if isinstance(commands, list):
            # Group by type for efficient encoding
            by_type = {}
            for cmd in commands:
                cmd_type = cmd.get("type", "unknown")
                by_type.setdefault(cmd_type, []).append(cmd)

            for cmd_type, cmds in by_type.items():
                msg = json.dumps({"type": cmd_type, "commands": cmds})
                self._ws.send(msg)
        else:
            # Single command
            msg = json.dumps({"type": commands.get("type", "unknown"), "data": commands})
            self._ws.send(msg)


# Factory function
def create_surface(width: int = 800, height: int = 600, title: str = "Actor Application"):
    """Create a WebSocket-driven HTML5 canvas surface.

    Returns a WebSocketSurface instance that can be used with
    WebSocketDriver. The WebSocket connection must be established
    separately via the JavaScript client.
    """
    return WebSocketSurface(width, height, title)