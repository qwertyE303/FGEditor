"""Virtools NMO object helpers: groups, placeholder 3D objects, Level scene.

The Level scene is stored in Level's ``0x80000000`` identifier.  It is a
Virtools state chunk whose data contains a fixed header, an object-ID list,
two subchunks per object, per-object scene flags, and a small identifier chain.
Keeping this codec here keeps ``nmo.py`` focused on the container.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .nmo import (
    CK_CLASS_3DENTITY,
    CK_CLASS_GROUP,
    StateChunk,
    u32,
)

# ``NULL_OBJECT`` lives in dataarray.py; defining it there would create a cycle.
NULL = 0xFFFFFFFF

GROUP_IDENT_ALL = 0x000FFFFF
OBJECT_IDENT_HIDDEN = 0x00000004
OBJECT_IDENT_MESH = 0x00004000
OBJECT_IDENT_ENTITY = 0x00100000
LEVEL_IDENT_SCENE = 0x80000000

SCENE_IDENT_NEWDATA = 0x00020000
SCENE_IDENT_TAIL = 0x00040000

DEFAULT_SCENE_FLAGS = 0x41


def _words(payload: bytes) -> list[int]:
    if len(payload) % 4:
        raise ValueError("payload is not a multiple of 4 bytes")
    if not payload:
        return []
    return list(struct.unpack(f"<{len(payload) // 4}I", payload))


def _pack_words(words: list[int]) -> bytes:
    if not words:
        return b""
    return struct.pack(f"<{len(words)}I", *words)


def encode_state_chunk(
    class_id: int,
    identifiers: list[tuple[int, bytes]],
    *,
    data_version: int = 10,
    chunk_version: int = 7,
    options: int = 0x08,
    trailing: bytes = b"",
) -> bytes:
    """Encode a state chunk with a flat identifier table."""
    words: list[int] = []
    for index, (ident, payload) in enumerate(identifiers):
        if len(payload) % 4:
            raise ValueError(f"identifier 0x{ident:X} payload is not dword aligned")
        payload_words = _words(payload)
        next_pos = len(words) + 2 + len(payload_words) if index + 1 < len(identifiers) else 0
        words.append(ident)
        words.append(next_pos)
        words.extend(payload_words)
    header = bytes([data_version, class_id, chunk_version, options])
    header += struct.pack("<I", len(words))
    return header + _pack_words(words) + trailing


def decode_group_members(chunk: bytes) -> list[int]:
    state = StateChunk.parse(chunk)
    payload = state.find(GROUP_IDENT_ALL)
    if payload is None or len(payload) < 4:
        return []
    count = u32(payload, 0)
    if 4 + count * 4 > len(payload):
        return []
    return list(struct.unpack_from(f"<{count}I", payload, 4))


def encode_group(members: list[int]) -> bytes:
    if not members:
        return encode_state_chunk(CK_CLASS_GROUP, [])
    payload = struct.pack("<I", len(members)) + struct.pack(f"<{len(members)}I", *members)
    return encode_state_chunk(CK_CLASS_GROUP, [(GROUP_IDENT_ALL, payload)])


def encode_placeholder_object() -> bytes:
    """Class 41 object: hidden, empty mesh, identity transform, no collision."""
    entity = struct.pack(
        "<2I12f",
        0x00000000,       # 3D entity flags (same as DepthTestCubes)
        0x00020001,       # moveable flags (same as DepthTestCubes)
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
    )
    identifiers = [
        (OBJECT_IDENT_HIDDEN, b""),
        (OBJECT_IDENT_MESH, struct.pack("<II", NULL, 0)),
        (OBJECT_IDENT_ENTITY, entity),
    ]
    return encode_state_chunk(CK_CLASS_3DENTITY, identifiers)


@dataclass
class SceneEntry:
    object_id: int
    subchunks: list[list[int]] = field(default_factory=lambda: [[0], [0]])
    flags: int = DEFAULT_SCENE_FLAGS


class LevelScene:
    """Parser/builder for the Level scene identifier payload."""

    def __init__(self) -> None:
        self.prefix = b"\xFF" * 8
        self.level_id = 0
        self.entries: list[SceneEntry] = []
        self.ids_tail: list[int] = []
        self.chunks_tail: list[int] = []
        self.tail_items: list[tuple[int, list[int]]] = []
        self.tail_fallback: list[int] | None = None
        self.header_class = 10
        self.header_version = 0x0007000A
        self.header_hasfile = 0
        self.header_mgr = 0
        self.old_n = 0
        self.old_sub_pos = 0

    @classmethod
    def from_payload(cls, payload: bytes) -> "LevelScene":
        if len(payload) < 40:
            raise ValueError("Level scene payload is too short")
        scene = cls()
        scene.prefix = payload[:8]
        size_m1, header_class, header_version, chunk_size, hasfile, idcount, chunkcount, mgr = (
            struct.unpack_from("<8I", payload, 8)
        )
        scene.header_class = header_class
        scene.header_version = header_version
        scene.header_hasfile = hasfile
        scene.header_mgr = mgr
        data_start = 40
        data_end = data_start + chunk_size * 4
        if data_end > len(payload):
            raise ValueError("Level scene data is truncated")
        data = _words(payload[data_start:data_end])
        tail_start = data_end
        ids_end = tail_start + idcount * 4
        chunks_end = ids_end + chunkcount * 4
        if chunks_end > len(payload):
            raise ValueError("Level scene tail is truncated")
        scene.ids_tail = _words(payload[tail_start:ids_end])
        scene.chunks_tail = _words(payload[ids_end:chunks_end])
        if len(data) < 5 or data[0] != SCENE_IDENT_NEWDATA:
            raise ValueError("unsupported Level scene layout")
        scene.level_id = data[2]
        n = data[3]
        id_count = data[4]
        if n != id_count or 5 + n > len(data):
            raise ValueError("invalid Level scene object count")
        ids = data[5:5 + n]
        pos = 5 + n
        if pos >= len(data):
            raise ValueError("Level scene is missing subchunk count")
        sub_count = data[pos]
        pos += 1
        subchunks: list[list[int]] = []
        for _ in range(sub_count):
            if pos >= len(data):
                raise ValueError("Level scene subchunks are truncated")
            size = data[pos]
            end = pos + 1 + size
            if end > len(data):
                raise ValueError("Level scene subchunk exceeds data")
            subchunks.append(data[pos:end])
            pos = end
        if 2 * n != sub_count:
            raise ValueError("Level scene subchunk count does not match objects")
        if pos + n > len(data):
            raise ValueError("Level scene flags are truncated")
        flags = data[pos:pos + n]
        pos += n
        scene.old_n = n
        scene.old_sub_pos = 5 + n
        for index, object_id in enumerate(ids):
            scene.entries.append(SceneEntry(
                object_id=object_id,
                subchunks=[subchunks[2 * index], subchunks[2 * index + 1]],
                flags=flags[index],
            ))
        tail = data[pos:]
        scene._parse_tail(data, pos)
        return scene

    def _parse_tail(self, data: list[int], tail_start: int) -> None:
        if tail_start >= len(data):
            self.tail_items = []
            self.tail_fallback = []
            return
        items: list[tuple[int, list[int]]] = []
        pos = tail_start
        while pos + 1 < len(data):
            ident = data[pos]
            next_pos = data[pos + 1]
            if next_pos == 0:
                end = len(data)
            else:
                if next_pos <= pos + 1 or next_pos > len(data):
                    items = []
                    break
                end = next_pos
            items.append((ident, data[pos + 2:end]))
            if next_pos == 0:
                pos = len(data)
                break
            pos = next_pos
        if items and pos >= len(data):
            self.tail_items = items
            self.tail_fallback = None
        else:
            self.tail_items = []
            self.tail_fallback = list(data[tail_start:])

    def to_payload(self) -> bytes:
        n = len(self.entries)
        data: list[int] = [SCENE_IDENT_NEWDATA, 0, self.level_id, n, n]
        data.extend(entry.object_id for entry in self.entries)
        data.append(2 * n)
        for entry in self.entries:
            first, second = entry.subchunks
            data.extend(first)
            data.extend(second)
        data.extend(entry.flags for entry in self.entries)

        tail_start = len(data)
        data[1] = tail_start
        if self.tail_items:
            pos = tail_start
            for index, (ident, payload) in enumerate(self.tail_items):
                next_pos = pos + 2 + len(payload) if index + 1 < len(self.tail_items) else 0
                data.append(ident)
                data.append(next_pos)
                data.extend(payload)
                pos = next_pos
        elif self.tail_fallback:
            data.extend(self.tail_fallback)
            if len(data) > 1:
                data[1] = tail_start

        delta = n - self.old_n
        chunks: list[int] = []
        for value in self.chunks_tail:
            if value != NULL and value >= self.old_sub_pos:
                chunks.append(value + delta)
            else:
                chunks.append(value)

        chunk_size = len(data)
        idcount = len(self.ids_tail)
        chunkcount = len(chunks)
        size_m1 = 8 + chunk_size + idcount + chunkcount - 1
        header = struct.pack(
            "<8I",
            size_m1,
            self.header_class,
            self.header_version,
            chunk_size,
            self.header_hasfile,
            idcount,
            chunkcount,
            self.header_mgr,
        )
        return (
            self.prefix + header + _pack_words(data)
            + _pack_words(self.ids_tail) + _pack_words(chunks)
        )

    def to_chunk(self, level_state: StateChunk) -> bytes:
        payload = self.to_payload()
        identifiers: list[tuple[int, bytes]] = []
        replaced = False
        for item in level_state.identifiers:
            if item.ident == LEVEL_IDENT_SCENE:
                identifiers.append((item.ident, payload))
                replaced = True
            else:
                identifiers.append((item.ident, item.payload))
        if not replaced:
            identifiers.append((LEVEL_IDENT_SCENE, payload))
        return encode_state_chunk(
            level_state.class_id,
            identifiers,
            data_version=level_state.data_version,
            chunk_version=level_state.chunk_version,
            options=level_state.options,
            trailing=level_state.trailing,
        )

    def update_entries(self, desired_ids: list[int]) -> None:
        """Keep existing entries for desired IDs, then append new IDs."""
        desired = list(dict.fromkeys(desired_ids))
        desired_set = set(desired)
        existing_by_id = {entry.object_id: entry for entry in self.entries}
        new_entries: list[SceneEntry] = []
        for entry in self.entries:
            if entry.object_id in desired_set:
                new_entries.append(entry)
        seen = {entry.object_id for entry in new_entries}
        for object_id in desired:
            if object_id in seen:
                continue
            entry = existing_by_id.get(object_id)
            if entry is None:
                entry = SceneEntry(object_id=object_id)
            new_entries.append(entry)
            seen.add(object_id)
        self.entries = new_entries
