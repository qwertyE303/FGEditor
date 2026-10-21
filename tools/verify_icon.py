"""Verify that a built exe embeds exactly the generated ICO frames.

The ICO frames are compared byte-for-byte with the exe's RT_ICON entries.
This is stricter than comparing rendered pixels and is what PyInstaller
normally preserves when copying the icon.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

try:
    import pefile
except Exception:  # pragma: no cover - build machine normally has it
    pefile = None


def ico_frames(path: Path) -> dict[int, bytes]:
    data = path.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    if reserved != 0 or kind != 1:
        raise ValueError("not an ICO file")
    frames = {}
    for index in range(count):
        width, height, _colors, _reserved, _planes, _bpp, size, offset = \
            struct.unpack_from("<BBBBHHII", data, 6 + 16 * index)
        frames[width or 256] = data[offset:offset + size]
    return frames


def exe_frames(path: Path) -> dict[int, bytes]:
    if pefile is None:
        raise RuntimeError("pefile is not installed")
    pe = pefile.PE(str(path))
    resources = pe.DIRECTORY_ENTRY_RESOURCE
    groups = [entry for entry in resources.entries if entry.id == 14]
    icons = next((entry for entry in resources.entries if entry.id == 3), None)
    if not groups or icons is None:
        raise ValueError("exe has no RT_GROUP_ICON/RT_ICON resources")
    group = groups[0]
    group_entry = group.directory.entries[0].directory.entries[0].data.struct
    group_data = pe.get_data(group_entry.OffsetToData, group_entry.Size)
    _reserved, _kind, count = struct.unpack_from("<HHH", group_data, 0)
    frames = {}
    for index in range(count):
        width, height, _colors, _reserved, _planes, _bpp, size, resource_id = \
            struct.unpack_from("<BBBBHHIH", group_data, 6 + 14 * index)
        icon = next((entry for entry in icons.directory.entries
                     if entry.id == resource_id), None)
        if icon is None:
            raise ValueError(f"missing RT_ICON id {resource_id}")
        entry = icon.directory.entries[0].data.struct
        frames[width or 256] = pe.get_data(entry.OffsetToData, entry.Size)
    return frames


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: verify_icon.py <exe> <ico>", file=sys.stderr)
        return 2
    exe = Path(argv[1])
    ico = Path(argv[2])
    expected = ico_frames(ico)
    actual = exe_frames(exe)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        print(f"icon size mismatch: missing={missing} extra={extra}", file=sys.stderr)
        return 1
    for size in sorted(expected):
        if expected[size] != actual[size]:
            print(f"icon frame {size}x{size} differs "
                  f"({len(expected[size])} vs {len(actual[size])} bytes)",
                  file=sys.stderr)
            return 1
    print(f"icon ok: {exe} ({len(expected)} frames, byte-for-byte)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
