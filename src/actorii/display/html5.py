"""HTML5 canvas display surface for ActorII.

Adapts the abstract DisplaySurface to produce actual HTML5 output
that can be rendered in a web browser, replacing the original
1993 Windows GDI-based rendering.

This module provides the HTML5Surface class which generates complete
HTML documents with embedded canvas elements and JavaScript rendering
runtime. It also provides Window, Widget, Button, Checkbox, and TextField
classes that produce HTML compatible with the canvas output.
"""

from __future__ import annotations


class DisplaySurface:
    """Abstract base class for display surfaces.

    The 1993 actor language was Windows-native with windowed GUI
    applications. This abstraction layer adapts those concepts to:
    - Linux Wayland native surfaces
    - HTML5 canvas rendering for web deployment
    - Cross-platform coordinate and event model
    """

    def __init__(self, width: int = 800, height: int = 600,
                 title: str = "Actor Application"):
        self.width = width
        self.height = height
        self.title = title
        self._pixels: list[list[str]] = [
            ["#FFFFFF"] * width for _ in range(height)
        ]
        self._objects: list = []
        self._event_queue: list[dict] = []
        self._root_window = None

    def clear(self, color: str = "#FFFFFF") -> None:
        """Clear the surface to a background color."""
        for y in range(self.height):
            for x in range(self.width):
                self._pixels[y][x] = color

    def set_pixel(self, x: int, y: int, color: str) -> None:
        """Set a pixel color at coordinates (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self._pixels[y][x] = color

    def get_pixel(self, x: int, y: int) -> str:
        """Get pixel color at coordinates (x, y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._pixels[y][x]
        return "#000000"

    def register_object(self, obj) -> None:
        """Register an actor object on this surface."""
        self._objects.append(obj)

    def unregister_object(self, obj) -> None:
        """Unregister an actor object."""
        self._objects.remove(obj)

    def post_event(self, event: dict) -> None:
        """Post an event to the event queue."""
        self._event_queue.append(event)

    def poll_event(self):
        """Poll the next event from the queue."""
        if self._event_queue:
            return self._event_queue.pop(0)
        return None

    def render(self) -> str:
        """Render the surface to output format.

        Returns HTML5 canvas-ready data or Wayland buffer reference.
        """
        raise NotImplementedError("Subclasses must implement render()")

    def flush(self) -> None:
        """Flush the surface (present frame to display)."""
        pass


class Window(DisplaySurface):
    """Window class adapted from 1993 actor language ACTOR*.CLS patterns.

    The original actor language had window classes (ABOUTWIN.CLS, etc.)
    for application windows. This modern adaptation provides:
    - Bounding rectangle with title
    - Child widget composition
    - Focus and z-order management
    - Close/minimize/maximize lifecycle
    """

    def __init__(self, x: int = 0, y: int = 0,
                 width: int = 800, height: int = 600,
                 title: str = "Actor Window",
                 surface: DisplaySurface = None):
        super().__init__(width, height, title)
        self.x = x
        self.y = y
        self._border_width = 2
        self._visible = True
        self._modal = False
        self._children: list[Widget] = []
        self._focus = None

        if surface is not None:
            self._pixels = surface._pixels
            self.width = surface.width
            self.height = surface.height

    def add_child(self, widget: 'Widget') -> None:
        """Add a child widget to this window."""
        self._children.append(widget)
        widget.parent = self
        self.register_object(widget)

    def remove_child(self, widget: 'Widget') -> None:
        """Remove a child widget."""
        self._children.remove(widget)
        widget.parent = None
        self.unregister_object(widget)

    def set_focus(self, widget: 'Widget') -> None:
        """Set keyboard focus to a widget."""
        self._focus = widget

    def handle_event(self, event: dict) -> bool:
        """Handle an event, propagating to children if unhandled.

        First tries focused widget, then children, then window-level handling.
        """
        # First try focused widget
        if self._focus and self._focus.handle_event(event):
            return True
        # Try children
        for child in self._children:
            if child.handle_event(event):
                return True
        # Window-level handling
        if event.get("type") == "keyboard":
            if event.get("key") == "Escape":
                self._visible = False
                return True
        return False

    def render(self) -> str:
        """Render the window and all children to HTML5-ready format.

        Generates an HTML div with embedded canvas and child widget renders.
        """
        canvas_id = f"window_{id(self)}"
        pixel_data = self._serialize_pixels()

        html = f'''<div id="{canvas_id}" \
style="position: relative; \
width: {self.width}px; \
height: {self.height}px; \
border: 2px solid #000000; \
background: #ffffff;">\
  <canvas id="{canvas_id}_canvas" \
  width="{self.width}" \
  height="{self.height}" \
  style="display: block;"></canvas>\
  <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">\
    {pixel_data}\
  </div>\
</div>'''

        # Add child widget renders
        for child in self._children:
            html += "\n<!-- Child: " + child.__class__.__name__ + " -->\n" + child.render()

        return html

    def _serialize_pixels(self) -> str:
        """Serialize pixel data as HTML-ready representation."""
        lines = []
        for row in self._pixels:
            hex_row = " ".join(pixel for pixel in row)
            lines.append(hex_row)
        return "\n".join(lines)


class Widget(DisplaySurface):
    """Base class for UI widgets (adapted from 1993 actor CONTROL.CLS,
    BUTTON.CLS, CHECKBOX.CLS patterns).

    Provides composable UI elements with position, state, and event handling.
    """

    def __init__(self, x: int = 0, y: int = 0,
                 width: int = 100, height: int = 30,
                 label: str = "",
                 surface: DisplaySurface = None):
        super().__init__(width, height, label)
        self.x = x
        self.y = y
        self._label = label
        self._visible = True
        self._enabled = True
        self._parent: 'Window' = None
        self._state: bool = False  # Checkbox state, button press state

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    def show(self) -> None:
        """Make this widget visible."""
        self._visible = True

    def hide(self) -> None:
        """Hide this widget."""
        self._visible = False

    def enable(self) -> None:
        """Enable this widget."""
        self._enabled = True

    def disable(self) -> None:
        """Disable this widget."""
        self._enabled = False

    def handle_event(self, event: dict) -> bool:
        """Handle an event on this widget. Returns True if handled.

        Subclasses override this to provide widget-specific event handling.
        """
        raise NotImplementedError

    def render(self) -> str:
        """Render the widget as HTML.

        Subclasses override this to provide widget-specific HTML output.
        """
        raise NotImplementedError


class Button(Widget):
    """Push button widget adapted from 1993 actor BUTTON.CLS patterns.

    The original actor language had push-button controls for
    user interaction. This modern adaptation provides:
    - Click detection via mouse events
    - Label text display
    - Push-state (returns True on click)
    """

    def __init__(self, x: int = 0, y: int = 0,
                 width: int = 100, height: int = 30,
                 label: str = "OK",
                 surface: DisplaySurface = None):
        super().__init__(x, y, width, height, label, surface)
        self._pressed = False

    def handle_event(self, event: dict) -> bool:
        """Handle mouse events on this button.

        Returns True if the event was handled.
        """
        if not self._visible or not self._enabled:
            return False

        event_type = event.get("type")

        if event_type == "mousebutton" and event.get("pressed"):
            mx, my = event.get("pos", (0, 0))
            # Check if click is within button bounds
            if (self.x <= mx <= self.x + self.width and
                self.y <= my <= self.y + self.height):
                self._pressed = True
                self.post_event({"type": "button_click", "source": self})
                return True

        if event_type == "mousebutton" and not event.get("pressed") and self._pressed:
            # Button release - fire action
            self._pressed = False
            self.post_event({"type": "button_action", "source": self,
                             "label": self._label})
            return True

        return False

    def render(self) -> str:
        """Render the button as HTML button element."""
        button_id = "btn_" + str(id(self))
        return '<button id="' + button_id + '" ' + \
            'style="position: absolute; left: ' + str(self.x) + 'px; ' + \
            'top: ' + str(self.y) + 'px; width: ' + str(self.width) + \
            'px; height: ' + str(self.height) + 'px; ' + \
            'font-family: sans-serif; font-size: 12px;">' + self._label + '</button>'


class Checkbox(Widget):
    """Check box widget adapted from 1993 actor CHECKBOX.CLS patterns."""

    def __init__(self, x: int = 0, y: int = 0,
                 width: int = 100, height: int = 30,
                 label: str = "",
                 surface: DisplaySurface = None):
        super().__init__(x, y, width, height, label, surface)
        self._checked = False

    def handle_event(self, event: dict) -> bool:
        """Handle mouse click events to toggle checkbox state."""
        if not self._visible or not self._enabled:
            return False

        event_type = event.get("type")

        if event_type == "mousebutton" and event.get("pressed"):
            mx, my = event.get("pos", (0, 0))
            if (self.x <= mx <= self.x + self.width and
                self.y <= my <= self.y + self.height):
                self._checked = not self._checked
                self.post_event({"type": "checkbox_toggle",
                                 "source": self,
                                 "checked": self._checked,
                                 "label": self._label})
                return True

        return False

    def render(self) -> str:
        """Render the checkbox as HTML input element."""
        checkbox_id = "chk_" + str(id(self))
        checked_attr = "checked" if self._checked else ""
        return '<input type="checkbox" id="' + checkbox_id + '" ' + \
            'style="position: absolute; left: ' + str(self.x) + 'px; ' + \
            'top: ' + str(self.y) + 'px;\" ' + checked_attr + '> ' + self._label + '<br>'


class TextField(Widget):
    """Text input field adapted from 1993 actor language GUI patterns."""

    def __init__(self, x: int = 0, y: int = 0,
                 width: int = 200, height: int = 25,
                 label: str = "",
                 surface: DisplaySurface = None):
        super().__init__(x, y, width, height, label, surface)
        self._text = ""
        self._max_chars: int = 255

    def handle_event(self, event: dict) -> bool:
        """Handle keyboard events for text input."""
        if not self._visible or not self._enabled:
            return False

        event_type = event.get("type")

        if event_type == "keyboard" and event.get("keydown"):
            key = event.get("key", "")
            if len(key) == 1:  # Printable character
                if len(self._text) < self._max_chars:
                    self._text += key
                    self.post_event({"type": "text_changed",
                                     "source": self,
                                     "text": self._text})
                    return True
            elif key == "backspace" and len(self._text) > 0:
                self._text = self._text[:-1]
                self.post_event({"type": "text_changed",
                                 "source": self,
                                 "text": self._text})
                return True

        return False

    def render(self) -> str:
        """Render the text field as HTML input element."""
        field_id = "txt_" + str(id(self))
        return '<input type="text" id="' + field_id + '" ' + \
            'style="position: absolute; left: ' + str(self.x) + 'px; ' + \
            'top: ' + str(self.y) + 'px; width: ' + str(self.width) + \
            'px; height: ' + str(self.height) + 'px; ' + \
            'font-family: monospace; font-size: 12px;\" ' + \
            'value="' + self._text + '">'


class HTML5Surface(DisplaySurface):
    """Concrete display surface that renders directly to HTML5.

    Adapts the abstract DisplaySurface to produce actual HTML5 output
    that can be rendered in a web browser, replacing the original
    1993 Windows GDI-based rendering.

    Generates a complete HTML document with embedded canvas element
    and JavaScript rendering runtime that handles drawing operations
    and event forwarding via WebSocket.
    """

    def __init__(self, width: int = 800, height: int = 600,
                 title: str = "Actor HTML5 App"):
        super().__init__(width, height, title)
        self._canvas_id = "actor_canvas_" + str(id(self))

    def render(self) -> str:
        """Return a fully-formed, mobile-ready HTML5 document.

        Builds an interactive Actor 4.1 Visual OOP Studio containing:
        * a responsive header and WebSocket connection status bar;
        * live 1993 Actor 4.1 demo apps (8-Queens, Mandelbrot Fractal, Sieve Bar Chart, Clock Window);
        * ObjectDraw tool palette (Pen stroke, Rect, Circle, Window, +Button, +Checkbox, +TextField);
        * a touch- and click-enabled ``<canvas>`` element with percentage-scaled widget overlay;
        * an interactive 1993 Class Browser (browsing all 340+ .CLS/.ACT files) and ActorVM compiler/runner.
        """
        pixel_js = self._generate_canvas_data()
        widgets_html = "\n".join(
            obj.render() for obj in self._objects if hasattr(obj, "render")
        )

        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>{self.title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 10px;
            background: #121629;
            color: #eaeaea;
        }}
        .container {{ max-width: {self.width + 24}px; margin: 0 auto; }}
        h1 {{ color: #e94560; text-align: center; font-size: 1.2rem; margin: 4px 0 8px; letter-spacing: 0.5px; }}
        .status-bar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 10px;
            background: #0f3460;
            border-radius: 6px;
            margin-bottom: 8px;
            font-size: 12px;
            gap: 8px;
            flex-wrap: wrap;
        }}
        .badge {{
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            background: #533483;
            font-size: 11px;
        }}
        .badge.connected {{ background: #1b9aaa; color: #fff; }}
        .section-label {{
            font-size: 11px;
            text-transform: uppercase;
            color: #8d99ae;
            margin: 6px 0 4px;
            font-weight: 700;
            letter-spacing: 0.6px;
        }}
        .controls {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(88px, 1fr));
            gap: 6px;
            margin-bottom: 8px;
        }}
        .btn {{
            background: #16213e;
            color: #e0e0e0;
            border: 1px solid #284b63;
            padding: 8px 6px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            touch-action: manipulation;
            text-align: center;
        }}
        .btn:active, .btn.active {{
            background: #e94560;
            color: #fff;
            border-color: #ff758f;
        }}
        .btn.app-btn {{
            background: #1d3557;
            border-color: #457b9d;
        }}
        .btn.app-btn:active, .btn.app-btn.active {{
            background: #1b9aaa;
            border-color: #80ed99;
        }}
        .toolbar-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
            background: #16213e;
            padding: 6px 10px;
            border-radius: 6px;
            font-size: 12px;
            flex-wrap: wrap;
        }}
        .toolbar-row input[type="color"] {{
            width: 32px;
            height: 26px;
            border: none;
            background: none;
            padding: 0;
            cursor: pointer;
        }}
        .stage-wrapper {{
            position: relative;
            width: 100%;
            border: 2px solid #457b9d;
            border-radius: 8px;
            background: #fff;
            margin-bottom: 10px;
            overflow: hidden;
        }}
        #{self._canvas_id} {{
            background: #fff;
            display: block;
            width: 100%;
            height: auto;
            touch-action: none;
        }}
        #widget-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            overflow: hidden;
        }}
        #widget-layer > * {{ pointer-events: auto; }}
        .tabs {{
            display: flex;
            gap: 6px;
            margin-bottom: 6px;
        }}
        .tab-btn {{
            flex: 1;
            padding: 8px;
            background: #16213e;
            color: #a8dadc;
            border: 1px solid #284b63;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
        }}
        .tab-btn.active {{
            background: #0f3460;
            color: #fff;
            border-color: #e94560;
        }}
        .panel {{
            display: none;
            background: #0d1321;
            border: 1px solid #1d3557;
            border-radius: 6px;
            padding: 8px;
            margin-bottom: 10px;
        }}
        .panel.active {{ display: block; }}
        .browser-controls {{
            display: flex;
            gap: 6px;
            margin-bottom: 6px;
        }}
        .browser-controls select {{
            flex: 1;
            padding: 6px;
            background: #16213e;
            color: #fff;
            border: 1px solid #457b9d;
            border-radius: 4px;
            font-size: 12px;
        }}
        pre.code-view, textarea.code-input {{
            width: 100%;
            height: 170px;
            margin: 0;
            padding: 8px;
            background: #080b12;
            color: #80ed99;
            border: 1px solid #284b63;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11.5px;
            overflow: auto;
            white-space: pre-wrap;
        }}
        .log {{
            height: 110px;
            overflow-y: auto;
            background: #080b12;
            border: 1px solid #1d3557;
            border-radius: 6px;
            padding: 8px;
            font-family: monospace;
            font-size: 11.5px;
            white-space: pre-wrap;
            color: #a8dadc;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.title} — Actor 4.1 Visual OOP Studio</h1>
        <div class="status-bar">
            <span id="coords-info">Mode: Freehand Pen — Drag on canvas</span>
            <span id="ws-status" class="badge">Connecting...</span>
        </div>

        <div class="section-label">1993 Actor 4.1 Live Applications</div>
        <div class="controls">
            <button class="btn app-btn" type="button" onclick="launchQueens()">👑 8-Queens</button>
            <button class="btn app-btn" type="button" onclick="launchFractal()">🌀 Fractal</button>
            <button class="btn app-btn" type="button" onclick="launchSieve()">📊 Prime Sieve</button>
            <button class="btn app-btn" type="button" onclick="launchClock()">🕒 Clock Win</button>
            <button class="btn app-btn" type="button" onclick="drawDemoScene()">🪟 Inspector</button>
        </div>

        <div class="section-label">ObjectDraw &amp; Widget Palette (Tap or Drag on Surface)</div>
        <div class="controls">
            <button class="btn active" id="tool-pen" type="button" onclick="setTool('pen')">✏️ Pen</button>
            <button class="btn" id="tool-rect" type="button" onclick="setTool('rect')">▭ Rect</button>
            <button class="btn" id="tool-circle" type="button" onclick="setTool('circle')">◯ Circle</button>
            <button class="btn" id="tool-window" type="button" onclick="setTool('window')">🗔 +Window</button>
            <button class="btn" type="button" onclick="addDemoWidget('button')">🔘 +Button</button>
            <button class="btn" type="button" onclick="addDemoWidget('checkbox')">☑️ +Check</button>
            <button class="btn" type="button" onclick="addDemoWidget('textfield')">⌨️ +Input</button>
            <button class="btn" type="button" onclick="clearSurface()">🗑️ Clear</button>
        </div>

        <div class="toolbar-row">
            <label>Color: <input type="color" id="draw-color" value="#e94560"></label>
            <label>Size:
                <select id="brush-size" style="background:#0f3460;color:#fff;border:1px solid #457b9d;border-radius:4px;padding:3px;">
                    <option value="2">2px</option>
                    <option value="4" selected>4px</option>
                    <option value="8">8px</option>
                    <option value="14">14px</option>
                </select>
            </label>
            <span id="active-app-hint" style="margin-left:auto;color:#ffb703;font-size:11px;">Tip: Switch tools or launch an Actor 4.1 app above!</span>
        </div>

        <div class="stage-wrapper">
            <canvas id="{self._canvas_id}" width="{self.width}" height="{self.height}"></canvas>
            <div id="widget-layer">{widgets_html}</div>
        </div>

        <div class="tabs">
            <button class="tab-btn active" id="tab-btn-log" type="button" onclick="switchTab('log')">📋 Console</button>
            <button class="tab-btn" id="tab-btn-browser" type="button" onclick="switchTab('browser')">📚 1993 Class Browser</button>
            <button class="tab-btn" id="tab-btn-vm" type="button" onclick="switchTab('vm')">⚡ ActorVM Runner</button>
        </div>

        <div class="panel active" id="panel-log">
            <div class="log" id="event-log">ActorII HTML5 Surface initialized.\n</div>
        </div>

        <div class="panel" id="panel-browser">
            <div class="browser-controls">
                <select id="actor-folder" onchange="loadActorFolder()">
                    <option value="actcls">actcls (195 Core Classes)</option>
                    <option value="actact">actact (23 Action Scripts)</option>
                    <option value="drawcls">drawcls (ObjectDraw GUI)</option>
                    <option value="oglcls">oglcls (ObjectGraphics)</option>
                    <option value="sqlcls">sqlcls (SQL Classes)</option>
                    <option value="vidcls">vidcls (Video App)</option>
                    <option value="actres">actres (Window Resources)</option>
                </select>
                <select id="actor-file" onchange="loadActorFile()">
                    <option value="OBJECT.CLS">OBJECT.CLS</option>
                </select>
            </div>
            <pre class="code-view" id="actor-source-view">Select a 1993 Actor 4.1 class or action file above to inspect its source...</pre>
        </div>

        <div class="panel" id="panel-vm">
            <textarea class="code-input" id="vm-code-input">class Counter:
    def __init__(self, start):
        x = 10 + 32
        return x

    def step(self, delta):
        y = 100 - 15
        return y</textarea>
            <div style="display:flex;gap:8px;margin-top:6px;">
                <button class="btn app-btn" style="flex:1;" type="button" onclick="runActorVM()">▶️ Compile &amp; Run on ActorVM</button>
            </div>
            <pre class="code-view" id="vm-output-view" style="height:120px;margin-top:6px;">Tap "Compile &amp; Run on ActorVM" to parse AST &amp; emit 1993 Actor bytecode.</pre>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('{self._canvas_id}');
        const ctx = canvas.getContext('2d');
        const widgetLayer = document.getElementById('widget-layer');
        const logEl = document.getElementById('event-log');
        const statusEl = document.getElementById('ws-status');
        const coordsEl = document.getElementById('coords-info');
        const hintEl = document.getElementById('active-app-hint');
        const colorInput = document.getElementById('draw-color');
        const sizeInput = document.getElementById('brush-size');
        const width = canvas.width;
        const height = canvas.height;

        let ws = null;
        let widgetCount = 0;
        let currentTool = 'pen';
        let activeApp = null;
        let clockTimer = null;
        let queensIndex = 0;
        let fractalView = {{ cx: -0.6, cy: 0.0, scale: 2.8 }};
        let isPointerDown = false;
        let startPt = null;
        let lastPt = null;
        let strokePoints = [];
        let savedCanvasSnapshot = null;

        function addLog(msg) {{
            const t = new Date().toLocaleTimeString();
            logEl.textContent += `[${{t}}] ${{msg}}\\n`;
            logEl.scrollTop = logEl.scrollHeight;
        }}

        function stopClock() {{
            if (clockTimer) {{
                clearInterval(clockTimer);
                clockTimer = null;
            }}
        }}

        function switchTab(tab) {{
            ['log', 'browser', 'vm'].forEach((name) => {{
                document.getElementById(`tab-btn-${{name}}`).classList.toggle('active', name === tab);
                document.getElementById(`panel-${{name}}`).classList.toggle('active', name === tab);
            }});
            if (tab === 'browser') {{
                loadActorFolder();
            }}
        }}

        function setTool(tool) {{
            currentTool = tool;
            activeApp = null;
            ['pen', 'rect', 'circle', 'window'].forEach((t) => {{
                const el = document.getElementById(`tool-${{t}}`);
                if (el) el.classList.toggle('active', t === tool);
            }});
            coordsEl.textContent = `Tool: ${{tool.toUpperCase()}} — Touch/drag on canvas`;
            hintEl.textContent = tool === 'window'
                ? 'Tap anywhere on canvas to spawn an Actor 4.1 Window!'
                : `Drawing with ${{tool}} (${{colorInput.value}})`;
        }}

        function initSurface() {{
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);
{pixel_js}
            drawBanner('ActorII Surface — Ready');
            bindExistingWidgets();
        }}

        function bindExistingWidgets() {{
            // Convert any initial px-positioned widgets to percentage layout & attach listeners
            Array.from(widgetLayer.children).forEach((el, idx) => {{
                el.style.left = `${{4 + idx * 18}}%`;
                el.style.top = '12%';
                el.style.zIndex = '10';
                attachWidgetEvents(el, el.textContent || el.value || `Widget #${{idx + 1}}`);
            }});
        }}

        function attachWidgetEvents(el, label) {{
            if (el.dataset.bound) return;
            el.dataset.bound = '1';
            el.addEventListener('click', (e) => {{
                e.stopPropagation();
                addLog(`Widget action: ${{label}}`);
                sendWs({{ type: 'post_event', event: {{ type: 'widget_click', label: label }} }});
                // Visual pulse feedback on canvas
                ctx.save();
                ctx.fillStyle = 'rgba(233, 69, 96, 0.18)';
                ctx.fillRect(0, 36, width, 28);
                ctx.fillStyle = '#0f3460';
                ctx.font = 'bold 14px monospace';
                ctx.fillText(`⚡ Message dispatched: send(Widget, #clicked, "${{label}}")`, 16, 55);
                ctx.restore();
            }});
        }}

        function drawBanner(text = '{self.title} — Actor 4.1 Visual Surface') {{
            ctx.fillStyle = '#0f3460';
            ctx.fillRect(0, 0, width, 36);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 15px sans-serif';
            ctx.fillText(text, 14, 24);
        }}

        function drawActorWindow(x, y, w, h, title) {{
            ctx.save();
            ctx.fillStyle = 'rgba(0,0,0,0.18)';
            ctx.fillRect(x + 5, y + 5, w, h);
            ctx.fillStyle = '#f8f9fa';
            ctx.fillRect(x, y, w, h);
            ctx.strokeStyle = '#0f3460';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = '#16213e';
            ctx.fillRect(x, y, w, 28);
            ctx.fillStyle = '#e94560';
            ctx.fillRect(x + w - 24, y + 5, 18, 18);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px sans-serif';
            ctx.fillText(title, x + 10, y + 19);
            ctx.fillText('×', x + w - 19, y + 18);
            ctx.restore();
        }}

        function drawDemoScene() {{
            stopClock();
            activeApp = 'inspector';
            ctx.fillStyle = '#f1f5f9';
            ctx.fillRect(0, 0, width, height);
            drawBanner('Actor 4.1 Object Inspector & Hierarchy (BROWSER.CLS)');

            drawActorWindow(30, 55, 350, 240, 'Class Hierarchy (ACTCLS)');
            const tree = [
                'Object (OBJECT.CLS)',
                ' ├─ Collection (COLLECT.CLS)',
                ' │   ├─ Array / List / Set',
                ' │   └─ Dictionary (DICT.CLS)',
                ' └─ WindowsObject (WINOBJ.CLS)',
                '     ├─ Window (WINDOW.CLS)',
                '     │   ├─ FractalWindow (FRACWIN.CLS)',
                '     │   └─ ClockWindow (CLOCKWIN.CLS)',
                '     └─ Control / Button / CheckBox'
            ];
            ctx.fillStyle = '#1e293b';
            ctx.font = '13px monospace';
            tree.forEach((line, i) => ctx.fillText(line, 46, 104 + i * 20));

            drawActorWindow(410, 55, 360, 240, 'Inspector: Actor[#Sam] (Sieve)');
            ctx.fillStyle = '#0f3460';
            ctx.font = 'bold 13px monospace';
            ctx.fillText('Class: Sieve (inherits Object)', 426, 105);
            ctx.fillStyle = '#334155';
            ctx.font = '12.5px monospace';
            ctx.fillText('Methods: #sieve(self, cnt | flags count)', 426, 130);
            ctx.fillText('Status : Ready in ActorVM Workspace', 426, 152);
            ctx.fillStyle = '#e94560';
            ctx.fillRect(426, 175, 150, 95);
            ctx.fillStyle = '#1b9aaa';
            ctx.fillRect(596, 175, 150, 95);
            ctx.fillStyle = '#ffffff';
            ctx.font = 'bold 13px sans-serif';
            ctx.fillText('Message Dispatch', 445, 220);
            ctx.fillText('send(self, #msg)', 448, 240);
            ctx.fillText('Multi-Surface', 626, 220);
            ctx.fillText('Wayland + HTML5', 618, 240);

            drawActorWindow(30, 320, 740, 250, 'Interactive Sketch & Widget Stage (Draw or Add Controls Here)');
            addLog('Rendered Actor 4.1 Inspector & Class Hierarchy scene');
            sendWs({{ type: 'get_surface_info' }});
        }}

        // 1. 8-Queens App (QUEEN1.ACT)
        function renderQueensBoard(n, board, solIdx, totalSols) {{
            stopClock();
            activeApp = 'queens';
            ctx.fillStyle = '#1e293b';
            ctx.fillRect(0, 0, width, height);
            drawBanner(`QUEEN1.ACT — ${{n}}-Queens Solver (Solution #${{solIdx + 1}} of ${{totalSols}}) — Tap board for next!`);

            const boardPx = Math.min(width - 80, height - 90);
            const cell = Math.floor(boardPx / n);
            const ox = Math.floor((width - cell * n) / 2);
            const oy = 52;

            for (let r = 0; r < n; r++) {{
                for (let c = 0; c < n; c++) {{
                    ctx.fillStyle = (r + c) % 2 === 0 ? '#f8fafc' : '#475569';
                    ctx.fillRect(ox + c * cell, oy + r * cell, cell, cell);
                }}
            }}
            ctx.strokeStyle = '#e94560';
            ctx.lineWidth = 3;
            ctx.strokeRect(ox - 2, oy - 2, cell * n + 4, cell * n + 4);

            ctx.font = `${{Math.floor(cell * 0.68)}}px sans-serif`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            for (let r = 0; r < n; r++) {{
                const c = board[r];
                if (c >= 0) {{
                    const cx = ox + c * cell + cell / 2;
                    const cy = oy + r * cell + cell / 2;
                    ctx.fillStyle = '#e94560';
                    ctx.beginPath();
                    ctx.arc(cx, cy, cell * 0.36, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText('♛', cx, cy + 2);
                }}
            }}
            ctx.textAlign = 'start';
            ctx.textBaseline = 'alphabetic';
            hintEl.textContent = `Showing 8-Queens solution #${{solIdx + 1}}/${{totalSols}} — Tap board for next solution!`;
        }}

        function launchQueens() {{
            stopClock();
            activeApp = 'queens';
            if (ws && ws.readyState === WebSocket.OPEN) {{
                sendWs({{ type: 'solve_queens', n: 8, index: queensIndex }});
                queensIndex++;
            }} else {{
                // Offline fallback solution
                const fallback = [0, 4, 7, 5, 2, 6, 1, 3];
                renderQueensBoard(8, fallback, 0, 92);
            }}
        }}

        // 2. Mandelbrot Fractal App (FRACWIN.CLS)
        function launchFractal(reset = true) {{
            stopClock();
            if (reset) fractalView = {{ cx: -0.6, cy: 0.0, scale: 2.8 }};
            activeApp = 'fractal';
            drawBanner(`FRACWIN.CLS — Mandelbrot Explorer (scale=${{fractalView.scale.toFixed(4)}}) — Tap fractal to zoom!`);
            const w = width;
            const h = height - 36;
            const img = ctx.createImageData(w, h);
            const data = img.data;
            const maxIter = 48;
            const aspect = w / h;
            for (let py = 0; py < h; py++) {{
                const y0 = fractalView.cy + ((py / h) - 0.5) * fractalView.scale;
                for (let px = 0; px < w; px++) {{
                    const x0 = fractalView.cx + ((px / w) - 0.5) * fractalView.scale * aspect;
                    let x = 0, y = 0, iter = 0;
                    while (x * x + y * y <= 4 && iter < maxIter) {{
                        const xt = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = xt;
                        iter++;
                    }}
                    const idx = (py * w + px) * 4;
                    if (iter === maxIter) {{
                        data[idx] = 15; data[idx + 1] = 23; data[idx + 2] = 42; data[idx + 3] = 255;
                    }} else {{
                        data[idx] = (iter * 19) % 255;
                        data[idx + 1] = (iter * 9 + 80) % 255;
                        data[idx + 2] = (iter * 37 + 140) % 255;
                        data[idx + 3] = 255;
                    }}
                }}
            }}
            ctx.putImageData(img, 0, 36);
            hintEl.textContent = 'Fractal active: Tap anywhere on the Mandelbrot canvas to zoom in 2.2x!';
            addLog(`Rendered FRACWIN.CLS Mandelbrot at (${{fractalView.cx.toFixed(3)}}, ${{fractalView.cy.toFixed(3)}})`);
        }}

        // 3. Eratosthenes Prime Sieve & Bar Chart (SIEVE.ACT + CHARTWIN.CLS)
        function renderSieveChart(res) {{
            stopClock();
            activeApp = 'sieve';
            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(0, 0, width, height);
            drawBanner(`SIEVE.ACT + CHARTWIN.CLS — ${{res.count}} Primes found in 0..${{res.limit}}`);

            drawActorWindow(30, 52, 740, 340, `Prime Distribution Bar Chart (0..${{res.limit}})`);
            const buckets = res.buckets || [];
            const maxCnt = Math.max(1, ...buckets.map(b => b.count));
            const chartX = 75, chartY = 105, chartW = 650, chartH = 235;
            const barW = Math.floor(chartW / Math.max(1, buckets.length)) - 14;
            const palette = ['#e94560', '#1b9aaa', '#ffb703', '#8338ec', '#3a86ff', '#fb5607', '#06d6a0', '#ef476f'];

            buckets.forEach((b, i) => {{
                const bh = Math.round((b.count / maxCnt) * (chartH - 30));
                const bx = chartX + i * (barW + 14);
                const by = chartY + chartH - bh;
                ctx.fillStyle = palette[i % palette.length];
                ctx.fillRect(bx, by, barW, bh);
                ctx.fillStyle = '#0f3460';
                ctx.font = 'bold 12px sans-serif';
                ctx.fillText(`${{b.count}}`, bx + Math.max(4, barW / 2 - 8), by - 6);
                ctx.fillStyle = '#334155';
                ctx.font = '11px monospace';
                ctx.fillText(b.label, bx, chartY + chartH + 16);
            }});

            drawActorWindow(30, 408, 740, 170, 'First Primes Discovered by Actor[#Sam]:sieve()');
            ctx.fillStyle = '#1e293b';
            ctx.font = '13px monospace';
            const primeStr = (res.primes || []).join(', ');
            const chunk = 78;
            for (let i = 0; i < 4; i++) {{
                ctx.fillText(primeStr.slice(i * chunk, (i + 1) * chunk), 48, 456 + i * 24);
            }}
            hintEl.textContent = `SIEVE.ACT found ${{res.count}} primes up to ${{res.limit}}`;
            addLog(`Executed SIEVE.ACT: ${{res.count}} primes <= ${{res.limit}}`);
        }}

        function launchSieve() {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                sendWs({{ type: 'run_sieve', limit: 400 }});
            }} else {{
                renderSieveChart({{
                    limit: 100, count: 25,
                    primes: [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97],
                    buckets: [{{label:'0-24',count:9}},{{label:'25-49',count:6}},{{label:'50-74',count:6}},{{label:'75-100',count:4}}]
                }});
            }}
        }}

        // 4. Clock Window App (CLOCKWIN.CLS)
        function launchClock() {{
            stopClock();
            activeApp = 'clock';
            function drawTick() {{
                ctx.fillStyle = '#0f172a';
                ctx.fillRect(0, 0, width, height);
                drawBanner('CLOCKWIN.CLS — Live 1993 Actor Clock Window');
                drawActorWindow(180, 65, 440, 490, 'ClockWindow (CLOCKWIN.CLS)');
                const cx = 400, cy = 295, r = 155;
                ctx.fillStyle = '#ffffff';
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.fill();
                ctx.lineWidth = 5;
                ctx.strokeStyle = '#0f3460';
                ctx.stroke();

                for (let h = 1; h <= 12; h++) {{
                    const ang = (h * Math.PI / 6) - Math.PI / 2;
                    const tx = cx + Math.cos(ang) * (r - 24);
                    const ty = cy + Math.sin(ang) * (r - 24);
                    ctx.fillStyle = '#16213e';
                    ctx.font = 'bold 16px sans-serif';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(String(h), tx, ty);
                }}

                const now = new Date();
                const sec = now.getSeconds();
                const min = now.getMinutes() + sec / 60;
                const hr = (now.getHours() % 12) + min / 60;

                function drawHand(angle, len, w, col) {{
                    ctx.save();
                    ctx.strokeStyle = col;
                    ctx.lineWidth = w;
                    ctx.lineCap = 'round';
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.lineTo(cx + Math.cos(angle) * len, cy + Math.sin(angle) * len);
                    ctx.stroke();
                    ctx.restore();
                }}
                drawHand(hr * Math.PI / 6 - Math.PI / 2, r * 0.52, 7, '#16213e');
                drawHand(min * Math.PI / 30 - Math.PI / 2, r * 0.72, 5, '#0f3460');
                drawHand(sec * Math.PI / 30 - Math.PI / 2, r * 0.82, 2.5, '#e94560');

                ctx.fillStyle = '#16213e';
                ctx.font = 'bold 20px monospace';
                ctx.textAlign = 'center';
                ctx.fillText(now.toLocaleTimeString(), cx, 495);
                ctx.textAlign = 'start';
                ctx.textBaseline = 'alphabetic';
            }}
            drawTick();
            clockTimer = setInterval(drawTick, 1000);
            hintEl.textContent = 'CLOCKWIN.CLS active — ticking every second.';
            addLog('Launched CLOCKWIN.CLS live clock window');
        }}

        function clearSurface() {{
            stopClock();
            activeApp = null;
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, width, height);
            widgetLayer.innerHTML = '';
            widgetCount = 0;
            drawBanner();
            sendWs({{ type: 'clear', color: '#FFFFFF' }});
            addLog('Cleared surface and widgets');
        }}

        function addDemoWidget(kind, customX = null, customY = null) {{
            widgetCount++;
            const x = customX !== null ? customX : (50 + ((widgetCount * 95) % 520));
            const y = customY !== null ? customY : (90 + ((widgetCount * 65) % 380));
            const label = kind === 'button'
                ? `ActorBtn #${{widgetCount}}`
                : (kind === 'checkbox' ? `Check #${{widgetCount}}` : `Input #${{widgetCount}}`);

            // Always render immediately on mobile stage using percentage coordinates so it works offline & online
            const leftPct = Math.max(2, Math.min(78, (x / width) * 100)).toFixed(1);
            const topPct = Math.max(8, Math.min(86, (y / height) * 100)).toFixed(1);
            let el;
            if (kind === 'button') {{
                el = document.createElement('button');
                el.textContent = label;
                el.style.cssText = `position:absolute;left:${{leftPct}}%;top:${{topPct}}%;padding:6px 12px;background:#16213e;color:#fff;border:2px solid #e94560;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,0.35);z-index:10;`;
            }} else if (kind === 'checkbox') {{
                el = document.createElement('label');
                el.style.cssText = `position:absolute;left:${{leftPct}}%;top:${{topPct}}%;background:rgba(22,33,62,0.95);color:#fff;padding:6px 10px;border:1px solid #1b9aaa;border-radius:6px;font-size:13px;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,0.35);z-index:10;`;
                el.innerHTML = `<input type="checkbox" checked> ${{label}}`;
            }} else {{
                el = document.createElement('input');
                el.type = 'text';
                el.placeholder = label;
                el.style.cssText = `position:absolute;left:${{leftPct}}%;top:${{topPct}}%;width:130px;padding:5px 8px;background:#fff;color:#111;border:2px solid #0f3460;border-radius:5px;font-family:monospace;font-size:12px;z-index:10;`;
            }}
            attachWidgetEvents(el, label);
            widgetLayer.appendChild(el);
            addLog(`Added ${{kind}} "${{label}}" at (${{x}}, ${{y}})`);

            sendWs({{
                type: 'render_widget',
                widget_type: kind,
                x: x,
                y: y,
                width: 130,
                height: 34,
                label: label,
                local_rendered: true
            }});
        }}

        function getCanvasCoords(clientX, clientY) {{
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {{
                x: Math.max(0, Math.min(width - 1, Math.round((clientX - rect.left) * scaleX))),
                y: Math.max(0, Math.min(height - 1, Math.round((clientY - rect.top) * scaleY)))
            }};
        }}

        canvas.addEventListener('pointerdown', (e) => {{
            e.preventDefault();
            const pt = getCanvasCoords(e.clientX, e.clientY);

            // If an interactive app is active on canvas, handle tap!
            if (activeApp === 'queens') {{
                launchQueens();
                return;
            }}
            if (activeApp === 'fractal') {{
                const aspect = width / (height - 36);
                fractalView.cx += ((pt.x / width) - 0.5) * fractalView.scale * aspect;
                fractalView.cy += (((pt.y - 36) / (height - 36)) - 0.5) * fractalView.scale;
                fractalView.scale /= 2.2;
                launchFractal(false);
                return;
            }}

            if (currentTool === 'window') {{
                drawActorWindow(Math.max(10, pt.x - 110), Math.max(42, pt.y - 20), 240, 160, `ActorWindow #${{++widgetCount}}`);
                addLog(`Spawned ActorWindow at (${{pt.x}}, ${{pt.y}})`);
                return;
            }}

            isPointerDown = true;
            startPt = pt;
            lastPt = pt;
            strokePoints = [[pt.x, pt.y]];
            savedCanvasSnapshot = ctx.getImageData(0, 0, width, height);

            if (currentTool === 'pen') {{
                const col = colorInput.value;
                const sz = parseInt(sizeInput.value, 10) || 4;
                ctx.fillStyle = col;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, sz / 2, 0, Math.PI * 2);
                ctx.fill();
            }}
            coordsEl.textContent = `Touch: (${{pt.x}}, ${{pt.y}}) [${{currentTool}}]`;
        }});

        canvas.addEventListener('pointermove', (e) => {{
            if (!isPointerDown || !lastPt) return;
            e.preventDefault();
            const pt = getCanvasCoords(e.clientX, e.clientY);
            const col = colorInput.value;
            const sz = parseInt(sizeInput.value, 10) || 4;

            if (currentTool === 'pen') {{
                ctx.strokeStyle = col;
                ctx.lineWidth = sz;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(lastPt.x, lastPt.y);
                ctx.lineTo(pt.x, pt.y);
                ctx.stroke();
                lastPt = pt;
                strokePoints.push([pt.x, pt.y]);
            }} else if (currentTool === 'rect' && savedCanvasSnapshot) {{
                ctx.putImageData(savedCanvasSnapshot, 0, 0);
                ctx.fillStyle = col + '33';
                ctx.strokeStyle = col;
                ctx.lineWidth = sz;
                ctx.fillRect(startPt.x, startPt.y, pt.x - startPt.x, pt.y - startPt.y);
                ctx.strokeRect(startPt.x, startPt.y, pt.x - startPt.x, pt.y - startPt.y);
            }} else if (currentTool === 'circle' && savedCanvasSnapshot) {{
                ctx.putImageData(savedCanvasSnapshot, 0, 0);
                const rx = Math.hypot(pt.x - startPt.x, pt.y - startPt.y);
                ctx.fillStyle = col + '33';
                ctx.strokeStyle = col;
                ctx.lineWidth = sz;
                ctx.beginPath();
                ctx.arc(startPt.x, startPt.y, rx, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }}
            coordsEl.textContent = `Drag: (${{pt.x}}, ${{pt.y}}) [${{currentTool}}]`;
        }});

        window.addEventListener('pointerup', () => {{
            if (!isPointerDown) return;
            isPointerDown = false;
            if (currentTool === 'pen' && strokePoints.length > 0) {{
                sendWs({{
                    type: 'draw_stroke',
                    points: strokePoints.slice(0, 64),
                    color: colorInput.value,
                    size: parseInt(sizeInput.value, 10) || 4
                }});
            }}
            strokePoints = [];
        }});

        // Class Browser & VM Runner functions
        function loadActorFolder() {{
            const folder = document.getElementById('actor-folder').value;
            sendWs({{ type: 'list_actor_files', folder: folder }});
        }}

        function loadActorFile() {{
            const folder = document.getElementById('actor-folder').value;
            const file = document.getElementById('actor-file').value;
            if (file) sendWs({{ type: 'read_actor_file', folder: folder, file: file }});
        }}

        function runActorVM() {{
            const code = document.getElementById('vm-code-input').value;
            sendWs({{ type: 'run_actor_vm', source: code }});
        }}

        function sendWs(payload) {{
            if (ws && ws.readyState === WebSocket.OPEN) {{
                ws.send(JSON.stringify(payload));
            }}
        }}

        function connectWs() {{
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = window.location.hostname || 'localhost';
            const url = `${{proto}}//${{host}}:8765/`;
            try {{
                ws = new WebSocket(url);
                ws.onopen = () => {{
                    statusEl.textContent = 'WS Connected';
                    statusEl.classList.add('connected');
                    addLog(`Connected to ${{url}}`);
                    sendWs({{ type: 'get_surface_info' }});
                    sendWs({{ type: 'list_actor_files', folder: 'actcls' }});
                }};
                ws.onmessage = (ev) => {{
                    try {{
                        const msg = JSON.parse(ev.data);
                        if (msg.type === 'surface_info') {{
                            addLog(`Surface: ${{msg.data.width}}x${{msg.data.height}} (${{msg.data.format}})`);
                        }} else if (msg.type === 'stroke_ack') {{
                            addLog(`Synced stroke (${{msg.count}} pts) to surface`);
                        }} else if (msg.type === 'queens_solution') {{
                            renderQueensBoard(msg.n, msg.board, msg.index, msg.total_solutions);
                            addLog(`8-Queens solution #${{msg.index + 1}}/${{msg.total_solutions}}: [${{msg.board.join(', ')}}]`);
                        }} else if (msg.type === 'sieve_result') {{
                            renderSieveChart(msg.data);
                        }} else if (msg.type === 'actor_files') {{
                            const sel = document.getElementById('actor-file');
                            sel.innerHTML = '';
                            (msg.files || []).forEach((f) => {{
                                const opt = document.createElement('option');
                                opt.value = f;
                                opt.textContent = f;
                                sel.appendChild(opt);
                            }});
                            if (msg.files && msg.files.length > 0) {{
                                sendWs({{ type: 'read_actor_file', folder: msg.folder, file: msg.files[0] }});
                            }}
                        }} else if (msg.type === 'actor_file_content') {{
                            document.getElementById('actor-source-view').textContent =
                                `/* === ${{msg.folder}}/${{msg.file}} (1993 Actor Professional 4.1) === */\\n\\n` + msg.content;
                        }} else if (msg.type === 'vm_result') {{
                            const out = [
                                `Parsed Classes: ${{JSON.stringify(msg.classes)}}`,
                                `Constants     : ${{JSON.stringify(msg.constants)}}`,
                                `Bytecode (${{msg.instructions.length}} instructions):`,
                                ...msg.instructions.map((ins, i) => `  [${{String(i).padStart(2, '0')}}] opcode=0x${{ins[0].toString(16).padStart(2, '0')}} operand=${{ins[1]}}`)
                            ].join('\\n');
                            document.getElementById('vm-output-view').textContent = out;
                            addLog(`Compiled ${{msg.classes.length}} class(es) -> ${{msg.instructions.length}} ActorVM opcodes`);
                        }} else if (msg.type === 'cleared') {{
                            addLog('Server surface buffer cleared');
                        }} else if (msg.type === 'rendered_widget') {{
                            // Already rendered locally via percentage positioning
                        }}
                    }} catch (err) {{
                        addLog(`WS parse error: ${{err.message}}`);
                    }}
                }};
                ws.onclose = () => {{
                    statusEl.textContent = 'Disconnected';
                    statusEl.classList.remove('connected');
                    setTimeout(connectWs, 3000);
                }};
            }} catch (err) {{
                addLog(`WS connection failed: ${{err.message}}`);
            }}
        }}

        initSurface();
        drawDemoScene();
        connectWs();
    </script>
</body>
</html>'''
        return html

    def _generate_canvas_data(self) -> str:
        """Generate canvas drawing commands from pixel data.

        Creates JavaScript ctx.fillRect commands for each non-white pixel.
        Optimized by skipping white background pixels.
        """
        commands: list[str] = []
        for y, row in enumerate(self._pixels):
            for x, color in enumerate(row):
                # Skip white background
                if color.upper() == "#FFFFFF":
                    continue
                # Convert shorthand colors
                hex_color = color.lstrip('#')
                if len(hex_color) == 3:
                    hex_color = ''.join(c + c for c in hex_color)
                commands.append(f"ctx.fillStyle = '#{hex_color}'; ctx.fillRect({x}, {y}, 1, 1);")
        js_lines = "\n            ".join(commands) if commands else "ctx.fillStyle = '#000000';"
        return "            " + js_lines + "\n"

    def flush(self) -> None:
        """Present the rendered frame."""
        pass


# Legacy export for backward compatibility
HTML5Surface = HTML5Surface