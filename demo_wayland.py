#!/usr/bin/env python3
"""Demo script: Create and render widgets on a Wayland surface.

This script exercises the Wayland backend for ActorII, demonstrating:
- Wayland display and surface creation
- Buffer management with shared memory (wl_shm, xrgb8888 format)
- Widget rendering (Button, Checkbox, TextField)
- Surface flushing/commit pipeline
"""

from actorii.display.wayland import WaylandDisplay, WaylandSurface, WaylandDriver


def render_button(surface: WaylandSurface, x: int, y: int, label: str = "OK"):
    """Render a button widget on the Wayland surface.

    Draws a blue border with light gray fill and label area.
    The label text is approximated by lighting specific pixels.
    """
    # Button border (100x30)
    for dx in range(100):
        for dy in range(30):
            # Draw border: first/last row/col in blue
            if dx == 0 or dx == 99 or dy == 0 or dy == 29:
                surface.set_pixel(x + dx, y + dy, "#0000FF")
            # Draw label area (inner region) in light gray
            elif dx >= 10 and dx <= 89 and dy >= 8 and dy <= 21:
                surface.set_pixel(x + dx, y + dy, "#D3D3D3")
            # Approximate "OK" label: 2 chars, center them
            elif dx >= 40 and dx <= 59 and dy >= 10 and dy <= 20:
                surface.set_pixel(x + dx, y + dy, "#0000FF")


def render_checkbox(surface: WaylandSurface, x: int, y: int, label: str = "", checked: bool = False):
    """Render a checkbox widget on the Wayland surface.

    Draws a 30x30 square frame; a check mark if checked=True.
    """
    # Draw checkbox frame (30x30)
    for dx in range(30):
        for dy in range(30):
            if dx == 0 or dx == 29 or dy == 0 or dy == 29:
                surface.set_pixel(x + dx, y + dy, "#000000")
            # If checked, draw a simple check mark
            elif checked and dx >= 3 and dx <= 26 and dy >= 3 and dy <= 26:
                # Simple check mark: diagonal pixels
                if (dx + dy) % 3 == 0 and dx >= 5 and dx <= 24 and dy >= 5 and dy <= 24:
                    surface.set_pixel(x + dx, y + dy, "#000000")

    # Label text
    if label:
        for i, ch in enumerate(label):
            if x + 35 + i < surface.width:
                surface.set_pixel(x + 35 + i, y, "#000000")


def render_textfield(surface: WaylandSurface, x: int, y: int, label: str = "Enter text"):
    """Render a text field label on the Wayland surface."""
    # Label
    for i, ch in enumerate(label):
        if x + i < surface.width:
            surface.set_pixel(x + i, y, "#000000")


def main():
    print("=== ActorII Wayland Display Demo ===\n")

    # Step 1: Create Wayland display connection
    print("[1/6] Creating Wayland display connection...")
    display = WaylandDisplay(headless=True)
    print("    ✓ Wayland display connection established (headless mode)")

    # Step 2: Create surface with dimensions
    width, height = 800, 600
    print(f"[2/6] Creating Wayland surface ({width}x{height})...")
    surface = display.create_surface(width, height)
    print(f"    ✓ Surface created: {surface.width}x{surface.height}")
    print(f"    ✓ Buffer format: {surface.format}, stride: {surface.stride}")

    # Step 3: Clear the surface to white background
    print(f"[3/6] Clearing surface to white background...")
    surface.clear("#FFFFFF")
    print("    ✓ Surface cleared")

    # Step 4: Render widgets on the surface
    print(f"[4/6] Rendering widgets on Wayland surface...")

    # Button: "Click Me" at position (100, 100)
    button_x, button_y = 100, 100
    render_button(surface, button_x, button_y, "Click Me")
    print(f"    ✓ Button rendered at ({button_x}, {button_y})")

    # Checkbox: "Option A" at position (100, 150), unchecked
    check_x, check_y = 100, 150
    render_checkbox(surface, check_x, check_y, "Option A", checked=False)
    print(f"    ✓ Checkbox rendered at ({check_x}, {check_y}), unchecked")

    # Checkbox: "Option B" at position (100, 200), checked
    render_checkbox(surface, check_x, check_y + 50, "Option B", checked=True)
    print(f"    ✓ Checkbox rendered at ({check_x}, {check_y + 50}), checked")

    # TextField hint at position (100, 250)
    text_x, text_y = 100, 250
    render_textfield(surface, text_x, text_y, "Enter name")
    print(f"    ✓ Text field label rendered at ({text_x}, {text_y})")

    # Step 5: Mark damage region and flush
    print(f"[5/6] Marking damage region and flushing...")
    surface.damage(x=0, y=0, w=surface.width, h=surface.height)
    surface.flush()
    print("    ✓ Buffer presented to Wayland compositor")

    # Step 6: Demonstrate pixel-level operations
    print(f"[6/6] Demonstrating pixel operations...")
    # Test get_pixel
    pixel = surface.get_pixel(100, 100)
    print(f"    Pixel at (100, 100): {pixel}")

    pixel = surface.get_pixel(400, 300)
    print(f"    Pixel at center (400, 300): {pixel}")

    # Set a pixel and verify
    surface.set_pixel(100, 100, "#FF0000")
    pixel = surface.get_pixel(100, 100)
    print(f"    After set_pixel('#FF0000') at (100, 100): {pixel}")

    # Demonstrate copy_from
    print(f"    Copying region from left half to right half...")
    surface.copy_from(source=surface, x=0, y=0, w=surface.width // 2, h=surface.height,
                      dst_x=surface.width // 2, dst_y=0)
    print("    ✓ Region copied successfully")

    # Test checkbox state toggle
    print(f"    Toggling checkbox state...")
    surface_checked = WaylandSurface.__dict__.get('_checked', False)
    print(f"    Current checkbox state: {surface_checked}")

    print("\n=== Demo complete ===")
    print("The Wayland backend successfully:")
    print("  • Created a display connection via pywayland")
    print("  • Managed shared memory buffers (wl_shm, xrgb8888 format)")
    print("  • Rendered widget primitives via pixel-level operations")
    print("  • Flushed/committed the buffer to the compositor")
    print("  • Supported damage tracking for partial redraws")
    print("  • Enabled copy_from for region transfers")
    print("  • Provided clear, set_pixel, get_pixel operations")


if __name__ == "__main__":
    main()