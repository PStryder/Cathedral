"""
Generate Cathedral extension icons.

Run: python generate_icons.py

Requires: pip install pillow
"""

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("Install Pillow: pip install pillow")
    exit(1)

def create_cathedral_icon(size: int) -> Image.Image:
    """Create a simple Cathedral icon."""
    # Create image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Colors
    primary = (99, 102, 241)  # Indigo
    secondary = (139, 92, 246)  # Purple
    accent = (236, 72, 153)  # Pink

    padding = size // 8
    width = size - (padding * 2)
    height = size - (padding * 2)

    # Draw stylized cathedral/arch shape
    center_x = size // 2

    # Main arch
    arch_width = int(width * 0.7)
    arch_height = int(height * 0.8)
    arch_left = center_x - arch_width // 2
    arch_top = padding + int(height * 0.1)

    # Draw pointed arch (gothic style)
    points = [
        (arch_left, size - padding),  # bottom left
        (arch_left, arch_top + arch_height // 2),  # left side
        (center_x, arch_top),  # top point
        (arch_left + arch_width, arch_top + arch_height // 2),  # right side
        (arch_left + arch_width, size - padding),  # bottom right
    ]

    draw.polygon(points, fill=primary)

    # Inner glow/window
    inner_padding = size // 6
    inner_points = [
        (arch_left + inner_padding, size - padding - inner_padding),
        (arch_left + inner_padding, arch_top + arch_height // 2 + inner_padding),
        (center_x, arch_top + inner_padding * 2),
        (arch_left + arch_width - inner_padding, arch_top + arch_height // 2 + inner_padding),
        (arch_left + arch_width - inner_padding, size - padding - inner_padding),
    ]

    draw.polygon(inner_points, fill=secondary)

    # Center light
    light_radius = size // 8
    light_center = (center_x, arch_top + arch_height // 3)
    draw.ellipse(
        [
            light_center[0] - light_radius,
            light_center[1] - light_radius,
            light_center[0] + light_radius,
            light_center[1] + light_radius
        ],
        fill=accent
    )

    return img


if __name__ == "__main__":
    sizes = [16, 48, 128]

    for size in sizes:
        icon = create_cathedral_icon(size)
        filename = f"icon{size}.png"
        icon.save(filename)
        print(f"Created {filename}")

    print("Done! Icons generated.")
