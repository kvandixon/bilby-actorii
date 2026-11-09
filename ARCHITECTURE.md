> [!WARNING]
> This is a planning document and should not be considered as a source of truth without verification.

# Actor Professional 4.1 Clean-Room Display Layer Architecture

## Overview

This document defines the clean-room reverse-engineered display architecture separating core language execution from display drivers for Actor Professional 4.1, targeting Wayland via pywayland/cffi and web canvas via HTML5/WebSockets.

The architecture consists of four layered components:

1. **Core Actor Language Runtime** - Pure OOP foundation with message passing, inheritance, and object model
2. **Display Driver Abstraction Layer** - Interface between runtime and concrete display backends
3. **Wayland Protocol Bindings** - Native Linux display surface via pywayland/cffi
4. **WebSocket-Mediated Canvas Rendering** - HTML5 browser-based rendering path

---

## 1. Core Actor Language Runtime

The existing `actorii.core` module provides the foundation:

- `ActorObject` - Base class with identity, encapsulation, message passing
- `ActorClass` - Meta-class for class definitions with inheritance
- `ActorInheritanceHierarchy` - Single-root inheritance management
- `ActorMessage` - Method dispatch representation

**Key primitives from 1993 archive:**
- `pNEW` - Object construction/Instantiation
- `SEND` / `MSG` - Message passing
- `CLASS` / `OBJ` - Class/object distinction
- `INHER`, `EXTENDS` - Single inheritance
- `VIRT`, `OVER` - Virtual/override methods
- `PROP`, `FUNC`, `PROC` - Properties, functions, procedures
- `DISP` - Dispatch table

No direct graphics dependencies. Pure computational substrate.

---

## 2. Display Driver Abstraction Layer

New module: `actorii.display.driver` — the critical separator between core runtime and concrete backends.

### Abstract Base Classes

```python
from abc import ABC, abstractmethod
from actorii.core import ActorObject

class DisplayDriver(ABC):
    """Abstract base for all display drivers.
    
    The 1993 actor language was Windows-native with GDI-based rendering.
    This abstraction layer adapts those concepts to modern display backends.
    """
    
    @abstractmethod
    def present(self, surface) -> None:
        """Present the rendered surface to the display hardware."""
        pass
    
    @abstractmethod
    def resize(self, width: int, height: int) -> None:
        """Handle surface resize events."""
        pass
    
    @abstractmethod
    def handle_event(self, event: dict) -> bool:
        """Process input events (keyboard, mouse, etc.)."""
        pass
    
    @abstractmethod
    def clear(self, color: str = "#FFFFFF") -> None:
        """Clear the surface to a background color."""
        pass


class GraphicsContext(ABC):
    """Abstract graphics context for drawing operations.
    
    Corresponds to 1993 GDC (Graphics Device Context) patterns from
    ACTMAIN.$ and window class definitions.
    """
    
    @abstractmethod
    def move_to(self, x: int, y: int) -> None:
        pass
    
    @abstractmethod
    def line_to(self, x: int, y: int) -> None:
        pass
    
    @abstractmethod
    def fill_rect(self, x: int, y: int, w: int, h: int, color: str) -> None:
        pass
    
    @abstractmethod
    def fill_poly(self, points: list[tuple[int, int]], color: str) -> None:
        pass
    
    @abstractmethod
    def set_font(self, font_spec: str) -> None:
        pass
    
    @abstractmethod
    def set_color(self, color: str) -> None:
        pass
```

### Surface Interface

```python
class Surface(ABC):
    """Abstract display surface representing a render target.
    
    The 1993 actor language had window surfaces (WIN*.CLS) and
    screen buffers. This abstraction adapts those concepts.
    """
    
    @property
    @abstractmethod
    def width(self) -> int:
        pass
    
    @property
    @abstractmethod
    def height(self) -> int:
        pass
    
    @property
    @abstractmethod
    def stride(self) -> int:
        pass
    
    @property
    @abstractmethod
    def format(self) -> str:
        pass  # e.g., "argb32", "rgb24"
    
    @abstractmethod
    def get_pixel(self, x: int, y: int) -> str:
        pass
    
    @abstractmethod
    def set_pixel(self, x: int, y: int, color: str) -> None:
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Present the completed frame."""
        pass
```

---

## 3. Wayland Protocol Bindings

New module: `actorii.display.wayland` — native Linux Wayland compositor integration via pywayland/cffi.

### Wayland Display Connection

```python
from pywayland.client.connection import Connection
from pywayland.client.display import Display
from pywayland.client.surface import Surface
from pywayland.protocol.wlr import *  # includes xdg_wm_base, etc.

class WaylandDisplay:
    """Manages a Wayland display connection."""
    
    def __init__(self):
        self.connection = Connection("wayland-0")
        self.display = Display(self.connection)
        self.surface = Surface(self.connection, self.display)
        self.xdg_wm_base = self._get_xdg_wm_base()
    
    def _get_xdg_wm_base(self):
        # Bind xdg_wm_base for window management
        ...
    
    def create_surface(self, width: int, height: int) -> WaylandSurface:
        return WaylandSurface(self, width, height)


class WaylandSurface:
    """Wayland buffer surface for actor application rendering."""
    
    def __init__(self, display: WaylandDisplay, width: int, height: int):
        self.display = display
        self.width = width
        self.height = height
        self.buffer = self._create_buffer(width, height)
        self.buffer_width = width
        self.buffer_height = height
    
    def _create_buffer(self, w: int, h: int) -> any:
        # Create a wl_buffer with proper format (xrgb8888 or rgba8888)
        ...
    
    def attach_buffer(self, buffer: any) -> None:
        self.buffer = buffer
        # wl_surface.attach(wl_surface, buffer)
    
    def damage(self, x: int = 0, y: int = 0, w: int = None, h: int = None) -> None:
        # Mark damaged region for redisplay
        ...
    
    def commit(self) -> None:
        # wl_surface.commit() - present the buffer
        ...
    
    def set_opaque_region(self, x: int, y: int, w: int, h: int) -> None:
        # Set opaque region for damage optimization
        ...
    
    def set_input_region(self, x: int, y: int, w: int, h: int) -> None:
        # Set input region for hit-testing
        ...
```

### Rendering Primitives via Wayland

The driver translates actor language drawing operations to Wayland protocol commands:

| Actor 1993 GDI Operation | Wayland Protocol Equivalent |
|-------------------------|----------------------------|
| `SetPixel(x, y, color)` | `wl_buffer` + pixel manipulation via shared memory |
| `LineTo(x1, y1, x2, y2)` | Render via Cairo/Evas on wl_surface |
| `FillRect(x, y, w, h, color)` | Render via Cairo/Evas on wl_surface |
| `SetBkColor(color)` | Set rendering context color |
| `TextOut(x, y, string)` | Render via Pango/Cairo on wl_surface |
| `StretchBlt(...)` | Scale buffer via EGL/Cairo |

---

## 4. WebSocket-Mediated Canvas Rendering

New module: `actorii.display.websocket` — HTML5 canvas rendering path where the actor application runs in a browser and communicates rendering commands via WebSocket.

### WebSocket Canvas Protocol

```
Actor Runtime (Python)  ──────▶  WebSocket  ◀───────  Browser (JavaScript)
         │                                 │
         ▼                                 ▼
   Render Commands                  Framebuffer / Canvas State
```

### WebSocket Message Types

| Direction | Message Type | Payload | Description |
|-----------|-------------|---------|-------------|
| `→` | `init` | `{width, height, title}` | Initialize canvas surface |
| `→` | `clear` | `{color}` | Clear the canvas |
| `→` | `pixel` | `{x, y, color}` | Set individual pixel |
| `→` | `rect` | `{x, y, w, h, color}` | Fill rectangle |
| `→` | `poly` | `{points, color}` | Fill polygon |
| `→` | `text` | `{x, y, text, font, color}` | Draw text string |
| `→` | `flush` | `{}` | Present frame (commit) |
| `←` | `event` | `{type, x, y, key, state}` | Input event from browser |
| `←` | `resize` | `{width, height}` | Surface resize notification |

### JavaScript Canvas Runtime (Browser Side)

```javascript
// Minimal WebSocket-driven canvas runtime
const canvas = document.getElementById('actor-canvas');
const ctx = canvas.getContext('2d');
const ws = new WebSocket(`ws://${location.host}/actor`);

canvas.width = window.innerWidth;
canvas.height = window.innerHeight - 50;

// Handle incoming render commands
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    switch (msg.type) {
        case 'init':
            canvas.width = msg.width;
            canvas.height = msg.height;
            break;
        case 'clear':
            ctx.fillStyle = msg.color;
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            break;
        case 'pixel':
            ctx.fillStyle = msg.color;
            ctx.fillRect(msg.x, msg.y, 1, 1);
            break;
        case 'rect':
            ctx.fillStyle = msg.color;
            ctx.fillRect(msg.x, msg.y, msg.w, msg.h);
            break;
        case 'poly':
            ctx.fillStyle = msg.color;
            ctx.beginPath();
            msg.points.forEach(([x, y]) => ctx.lineTo(x, y));
            ctx.closePath();
            ctx.fill();
            break;
        case 'text':
            ctx.fillStyle = msg.color;
            ctx.font = msg.font || '12px sans-serif';
            ctx.fillText(msg.text, msg.x, msg.y);
            break;
        case 'flush':
            // Frame presented - could trigger redraw or notify parent
            break;
    }
};

// Forward input events
canvas.addEventListener('click', (e) => {
    const rect = canvas.getBoundingClientRect();
    ws.send(JSON.stringify({
        type: 'event',
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        type: 'mousebutton',
        pressed: true
    }));
});
```

---

## 5. Driver Integration Plan

### Phase 1: Abstraction Layer (Weeks 1-2)

1. Create `actorii/display/driver.py` with `DisplayDriver`, `Surface`, `GraphicsContext` abstract base classes
2. Migrate existing `HTML5Surface` from `display/__init__.py` to implement `Surface` ABC
3. Ensure all existing display classes (`Window`, `Widget`, `Button`, `Checkbox`, `TextField`) depend on the abstraction, not concretions
4. Test that existing tests still pass with the new abstraction layer

### Phase 2: Wayland Backend (Weeks 3-5)

1. Install `pywayland` package
2. Create `actorii/display/wayland.py` with `WaylandDisplay`, `WaylandSurface` classes
3. Implement `WaylandDriver(DisplayDriver)` — concrete Wayland backend
4. Translate `GraphicsContext` operations to Cairo/Evas rendering on wl_surface
5. Implement pixel buffer management with proper Wayland format (xrgb8888)
6. Integrate with xdg_wm_base for window decoration and management
7. Test with actual Wayland compositor (sway, gnome-shell, etc.)

### Phase 3: WebSocket Canvas Backend (Weeks 6-8)

1. Create `actorii/display/websocket.py` with `WebSocketDriver`, `WebSocketSurface` classes
2. Implement WebSocket protocol handler (send/receive message types defined above)
3. Create JavaScript canvas runtime for browser deployment
4. Test HTML5 output generation and WebSocket communication
5. Optimize command batching for performance (reduce per-pixel WebSocket messages)

### Phase 4: Integration & Optimization (Weeks 9-10)

1. Add driver selection mechanism (environment variable or config)
2. Implement double-buffering and damage tracking across both backends
3. Add font rendering support (Pango for Wayland, Canvas 2D for WebSocket)
4. Implement input event pipeline (keyboard, mouse, touch)
5. Performance benchmarking and optimization
6. Documentation and examples

---

## 6. Interface Specification

### Driver Registration Protocol

```python
# Core runtime discovers available drivers via entry points or config

# pyproject.toml [project.scripts] or [tool.hatch.plugins]
# actorii-wayland = "actorii.display.wayland:main"
# actorii-websocket = "actorii.display.websocket:main"

# Or via environment variable
ACTORII_DISPLAY_DRIVER=wayland  # or "websocket"

# Default: auto-detect (wayland preferred, fallback to websocket)
```

### Surface Lifecycle

```
+----------------------+
|  Actor Language VM   |
|  (message passing)   |
+----------+-----------+
           |
           v
+----------+-----------+     +----------------------+
|  Display Driver      |     |  WebSocket Driver    |
|  (Abstract Base)    |     |  (Browser Deployment)|
+----------+-----------+     +----------+-----------+
           |                          |
           v                          v
+----------+-----------+  WebSocket  +----------+-----------+
|   Wayland Backend    |◄────────────▶|  HTML5 Canvas      |
|  (pywayland/cffi)   |  Protocol   |  (JavaScript)      |
+-----------------------+-------------+--------------------+
```

### Event Pipeline

```
Input Event
    ↓
[Wayland: wl_keyboard.wl_surface_keyboard
     wl_pointer.enter/leave
     wl_pointer.frame]
     ↓
[WebSocket: JSON {type, x, y, key, state}]
    ↓
Actor Language Event Handler
    ↓
Message passing via ActorObject.dispatch()
    ↓
Side effects (state changes, new events posted)
```

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                           CORE LAYER                             │
│  actorii.core → ActorObject, ActorClass, ActorMessage           │
│  Pure OOP foundation, no graphics dependencies                   │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DISPLAY ABSTRACTION LAYER                     │
│  actorii.display.driver → DisplayDriver, Surface, GraphicsContext│
│  Abstract interfaces, no concrete backend knowledge             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Wayland       │ │   WebSocket     │ │   Hybrid/Other  │
│  pywayland/cffi │ │  HTML5/WS       │ │   backends      │
│  actorii.wayland│ │ actorii.ws      │ │  (Qt, X11, etc)│
│  WaylandDisplay │ │ WebSocketSurface│ │               │
│  WaylandSurface │ │ (JavaScript)   │ │               │
│  Cairo rendering│ │ WebSocket CLI   │ │               │
│  xdg_wm_base    │ │   Protocol      │ │               │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## 8. Design Rationale

### Clean-Separation Guarantees

1. **Core → Abstraction**: Core language never references Wayland, WebSocket, or HTML. It only knows `ActorObject`, `ActorClass`, and `ActorMessage`. The abstraction layer is the sole bridge.

2. **Abstraction → Backend**: The display abstraction layer knows about concrete backend protocols (Wayland shell/compositor, WebSocket wire format) but the core does not.

3. **Implementation Isolation**: Wayland and WebSocket implementations can evolve independently. A bug in Wayland pixel handling cannot affect WebSocket command framing.

### Preserved 1993 Actor Language Semantics

| 1993 Feature | 2026 Adaptation |
|-------------|-----------------|
| Windows GDC / HDC | Wayland wl_surface + Cairo, or HTML5 Canvas 2D |
| GetPixel/SetPixel | Surface pixel buffer access |
| TextOut via system font | Pango (Wayland) / Canvas 2D (Web) |
| Window classes (WIN*.CLS) | xdg_toplevel / HTML5 window div |
| Message events (BUTTON.CLS, etc.) | Input pipeline → ActorMessage dispatch |
| Screen buffer present (FlushBlt) | `wl_surface.commit()` / WebSocket `flush` |

### Performance Considerations

- **Wayland**: Direct pixel buffer sharing via wl_shm (shared memory) — minimal copy overhead
- **WebSocket**: Batch rendering commands instead of per-pixel messages; use JSON compression for dense scenes
- **Damage tracking**: Region-based invalidation (wl_surface.damage() + WebSocket `damage` messages) to avoid full-frame redraws
- **Font caching**: Pre-rendered glyph textures for Wayland; Canvas 2D font rendering for Web (acceptable for development/ prototyping)

---

## 9. Implementation Checklist

### Abstraction Layer (Phase 1)
- [ ] Create `actorii/display/driver.py` with ABCs
- [ ] Migrate `HTML5Surface` to implement `Surface` ABC
- [ ] Ensure `Window`, `Widget`, `Button`, `Checkbox`, `TextField` use dependency injection
- [ ] Verify all 14 existing tests pass

### Wayland Backend (Phase 2)
- [ ] `pip install pywayland`
- [ ] Create `actorii/display/wayland.py`
- [ ] Implement `WaylandDriver(DisplayDriver)`
- [ ] Implement pixel buffer management (wl_shm)
- [ ] Integrate with xdg_wm_base
- [ ] Test with running Wayland compositor

### WebSocket Backend (Phase 3)
- [ ] Create `actorii/display/websocket.py`
- [ ] Implement WebSocket protocol handler
- [ ] Create JavaScript client runtime
- [ ] Test HTML5 output generation

### Polish (Phase 4)
- [ ] Damage tracking implementation
- [ ] Font rendering (Pango/Canvas 2D)
- [ ] Input event pipeline (keyboard/mouse)
- [ ] Driver auto-detection and selection
- [ ] Performance benchmarks
- [ ] Documentation and examples

---

*Design generated from clean-room reverse engineering of Actor Professional 4.1 (1993)
Whitewater Group / Genesis Development Systems.*
*Target: Linux Wayland via pywayland/cffi + HTML5/WebSockets deployment.*