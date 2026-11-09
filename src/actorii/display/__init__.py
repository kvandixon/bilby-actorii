"""Display surface module for Actor language.

Handles the clean-room reverse-engineered display architecture from
the 1993 actor language, adapting Windows-native GUI concepts to
modern Linux Wayland and HTML5 canvas rendering.

Key architectural elements from the 1993 archive:
- Window classes (WIN*.CLS) for UI composition
- Behavior classes (BEHAVIOR.CLS) for event handling
- Control classes (CONTROL.CLS, BUTTON.CLS, etc.) for composable UI
- ACTRES resource files for asset management
- Message-passing event dispatch

Display backends:
- Wayland via pywayland/cffi (native Linux)
- HTML5 via WebSocket (browser deployment)
"""

from __future__ import annotations

# Core abstract interfaces - always available
from actorii.display.driver import (
    DisplayDriver,
    Surface,
    GraphicsContext,
)

# DisplaySurface is the base class defined in html5 module
from actorii.display.html5 import DisplaySurface, HTML5Surface, Window, Widget, Button, Checkbox, TextField

# Try to import Wayland backend; make graceful if not available
try:
    from actorii.display.wayland import WaylandSurface, WaylandDisplay, WaylandDriver
except ImportError:
    WaylandSurface = None  # type: ignore
    WaylandDisplay = None  # type: ignore
    WaylandDriver = None  # type: ignore

# WebSocket/HTML5-over-WebSocket backend
from actorii.display.websocket import WebSocketSurface, WebSocketDriver, create_surface

# Export all symbols
__all__ = [
    # Abstract interfaces
    "DisplayDriver",
    "Surface",
    "GraphicsContext",
    "DisplaySurface",
    # HTML5/backend widgets
    "HTML5Surface",
    "Window",
    "Widget",
    "Button",
    "Checkbox",
    "TextField",
    # WebSocket surface
    "WebSocketSurface",
    "WebSocketDriver",
    "create_surface",
    # Wayland backend (may be None if pywayland not installed)
    "WaylandSurface",
    "WaylandDisplay",
    "WaylandDriver",
]