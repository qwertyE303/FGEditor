"""Generate the FGEditor application icon.

Structure follows build/gen/z256.png: two balls joined by dipole field lines,
with no plus/minus glyphs.  The left ball uses Ball_Wood.bmp and the right
ball uses Ball_Stone.bmp; no tint is applied.

The ICO uses BMP frames for 16-128 px and PNG for 192/256 px, which is the
most compatible layout for the Windows shell at small sizes.
"""

from __future__ import annotations

import argparse
import io
import math
import struct
import sys
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
CX, CY = S / 2, S * 0.52
DX, R = S * 0.175, S * 0.145

WOOD = Path(r"D:\Ballance\Textures\Ball_Wood.bmp")
STONE = Path(r"D:\Ballance\Textures\Ball_Stone.bmp")
WOOD_FALLBACK = ((214, 166, 112), (94, 57, 28))
STONE_FALLBACK = ((206, 212, 202), (86, 94, 84))

FIELD_EDGE = (12, 52, 92, 225)
FIELD_CORE = (126, 212, 255)
RIM = (10, 32, 58, 210)
LOOPS = ((0.06, 235), (0.15, 205), (0.26, 175), (0.40, 145), (0.58, 115))
LIGHT = (-0.42, -0.55, 0.72)


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _sample_bilinear(texture: Image.Image, u: float, v: float):
    width, height = texture.size
    x = (u % 1.0) * width - 0.5
    y = (v % 1.0) * height - 0.5
    x0 = math.floor(x)
    y0 = math.floor(y)
    fx = x - x0
    fy = y - y0
    x0 %= width
    y0 %= height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    p00 = texture.getpixel((x0, y0))
    p10 = texture.getpixel((x1, y0))
    p01 = texture.getpixel((x0, y1))
    p11 = texture.getpixel((x1, y1))
    return tuple(
        int(p00[i] * (1 - fx) * (1 - fy) + p10[i] * fx * (1 - fy) +
            p01[i] * (1 - fx) * fy + p11[i] * fx * fy)
        for i in range(3)
    )


def render_sphere(size: int, texture: Image.Image | None,
                  fallback: tuple[tuple[int, int, int], tuple[int, int, int]]) -> Image.Image:
    radius = size / 2 - 1
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    norm = math.sqrt(sum(v * v for v in LIGHT))
    lx, ly, lz = (v / norm for v in LIGHT)
    for y in range(size):
        ny = (y + 0.5 - size / 2) / radius
        for x in range(size):
            nx = (x + 0.5 - size / 2) / radius
            d2 = nx * nx + ny * ny
            if d2 > 1.0:
                continue
            nz = math.sqrt(1.0 - d2)
            if texture is not None:
                u = 0.5 + math.atan2(nx, nz) / (2 * math.pi)
                v = 0.5 - math.asin(max(-1.0, min(1.0, ny))) / math.pi
                base = _sample_bilinear(texture, u * 1.7, v * 1.7)
            else:
                base = lerp(fallback[0], fallback[1],
                            min(1.0, math.sqrt(d2)) ** 0.85)
            diff = max(0.0, nx * lx + ny * ly + nz * lz)
            shade = 0.40 + 0.80 * diff
            red, green, blue = base[0] * shade, base[1] * shade, base[2] * shade
            spec = diff ** 28
            red += 120 * spec
            green += 110 * spec
            blue += 95 * spec
            alpha = 255
            if d2 > 0.94:
                alpha = int(255 * (1.0 - d2) / 0.06)
            px[x, y] = (min(255, int(red)), min(255, int(green)),
                        min(255, int(blue)), alpha)
    return img


def sweep_pts(center, radius, a0, a1):
    out = []
    for direction in (1, -1):
        span = ((a1 - a0) % (2 * math.pi)) if direction == 1 else -((a0 - a1) % (2 * math.pi))
        points = [
            (center[0] + radius * math.cos(a0 + span * i / 160),
             center[1] + radius * math.sin(a0 + span * i / 160))
            for i in range(161)
        ]
        angle = (a0 + span / 2) % (2 * math.pi)
        out.append((points, angle))
    return out


def field_lines(img: Image.Image, cx, cy, dx, radius, sign):
    draw = ImageDraw.Draw(img)
    for offset, _alpha in LOOPS:
        loop_y = cy + sign * offset * S
        loop_r = math.hypot(dx, offset * S)
        a0 = math.atan2(cy - loop_y, -dx)
        a1 = math.atan2(cy - loop_y, dx)
        want = (-math.pi / 2 if sign < 0 else math.pi / 2) % (2 * math.pi)
        best = None
        for points, angle in sweep_pts((cx, loop_y), loop_r, a0, a1):
            diff = min(abs(angle - want), 2 * math.pi - abs(angle - want))
            if best is None or diff < best[0]:
                best = (diff, points)
        points = best[1]
        segment = []
        for point in points:
            if (math.hypot(point[0] - (cx - dx), point[1] - cy) < radius * 1.0 or
                    math.hypot(point[0] - (cx + dx), point[1] - cy) < radius * 1.0):
                if len(segment) > 1:
                    draw.line(segment, fill=FIELD_EDGE, width=int(S * 0.019), joint="curve")
                    segment = []
            else:
                segment.append(point)
        if len(segment) > 1:
            draw.line(segment, fill=FIELD_EDGE, width=int(S * 0.019), joint="curve")

    core = Image.new("RGBA", img.size, (0, 0, 0, 0))
    core_draw = ImageDraw.Draw(core)
    for offset, alpha in LOOPS:
        loop_y = cy + sign * offset * S
        loop_r = math.hypot(dx, offset * S)
        a0 = math.atan2(cy - loop_y, -dx)
        a1 = math.atan2(cy - loop_y, dx)
        want = (-math.pi / 2 if sign < 0 else math.pi / 2) % (2 * math.pi)
        best = None
        for points, angle in sweep_pts((cx, loop_y), loop_r, a0, a1):
            diff = min(abs(angle - want), 2 * math.pi - abs(angle - want))
            if best is None or diff < best[0]:
                best = (diff, points)
        points = best[1]
        segment = []
        for point in points:
            if (math.hypot(point[0] - (cx - dx), point[1] - cy) < radius * 0.98 or
                    math.hypot(point[0] - (cx + dx), point[1] - cy) < radius * 0.98):
                if len(segment) > 1:
                    core_draw.line(segment, fill=(*FIELD_CORE, alpha),
                                   width=int(S * 0.010), joint="curve")
                    segment = []
            else:
                segment.append(point)
        if len(segment) > 1:
            core_draw.line(segment, fill=(*FIELD_CORE, alpha),
                           width=int(S * 0.010), joint="curve")
    img.alpha_composite(core)


def draw_rim(img: Image.Image, cx, cy, radius):
    draw = ImageDraw.Draw(img)
    extra = int(S * 0.006)
    draw.ellipse((cx - radius - extra, cy - radius - extra,
                  cx + radius + extra, cy + radius + extra), fill=RIM)


def load_texture(path: Path):
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def build_master(wood_texture, stone_texture) -> Image.Image:
    diameter = int(R * 2)
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    for sign in (-1, 1):
        field_lines(img, CX, CY, DX, R, sign)

    draw_rim(img, CX - DX, CY, R)
    draw_rim(img, CX + DX, CY, R)

    wood = render_sphere(diameter, wood_texture, WOOD_FALLBACK)
    stone = render_sphere(diameter, stone_texture, STONE_FALLBACK)
    img.alpha_composite(wood, (int(CX - DX - R), int(CY - R)))
    img.alpha_composite(stone, (int(CX + DX - R), int(CY - R)))
    return img


def _bmp_frame(image: Image.Image) -> bytes:
    width, height = image.size
    rgba = image.convert("RGBA")
    xor = bytearray()
    mask = bytearray()
    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b, a = rgba.getpixel((x, y))
            xor += bytes((b, g, r, a))
        mask_row = bytearray((width + 31) // 32 * 4)
        for x in range(width):
            if rgba.getpixel((x, y))[3] == 0:
                mask_row[x // 8] |= 0x80 >> (x % 8)
        mask += mask_row
    header = struct.pack(
        "<IiiHHIIiiII", 40, width, height * 2, 1, 32, 0,
        len(xor) + len(mask), 0, 0, 0, 0)
    return header + bytes(xor) + bytes(mask)


def _png_frame(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def write_ico(path: Path, frames: list[tuple[int, Image.Image]]) -> None:
    entries = []
    blobs = []
    offset = 6 + 16 * len(frames)
    for size, image in frames:
        blob = _bmp_frame(image) if size <= 128 else _png_frame(image)
        width_byte = 0 if size >= 256 else size
        height_byte = 0 if size >= 256 else size
        entries.append(struct.pack(
            "<BBBBHHII", width_byte, height_byte, 0, 0, 1, 32, len(blob), offset))
        blobs.append(blob)
        offset += len(blob)
    header = struct.pack("<HHH", 0, 1, len(frames))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + b"".join(entries) + b"".join(blobs))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[2] / "build" / "gen" / "FGEditor.ico")
    parser.add_argument("--preview", type=Path, default=None)
    parser.add_argument("--wood", type=Path, default=WOOD)
    parser.add_argument("--stone", type=Path, default=STONE)
    args = parser.parse_args()

    wood = load_texture(args.wood)
    stone = load_texture(args.stone)
    if wood is None:
        print(f"warning: wood texture not found: {args.wood}; using fallback", file=sys.stderr)
    if stone is None:
        print(f"warning: stone texture not found: {args.stone}; using fallback", file=sys.stderr)

    master = build_master(wood, stone)
    sizes = (16, 24, 32, 48, 64, 128, 192, 256)
    frames = [(size, master.resize((size, size), Image.LANCZOS)) for size in sizes]
    write_ico(args.out, frames)
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        master.resize((256, 256), Image.LANCZOS).save(args.preview)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
