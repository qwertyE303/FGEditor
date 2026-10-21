"""Virtools NMO (NeMo) container reader/writer.

Only the subset needed by the FlexGravity editor is implemented.  The layout
and CRC/compression rules were verified against Virtools 2.1 / Ballance NMO
files produced by Virtools 3.5.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

NEMO_MAGIC = b"Nemo Fi\x00"
HEADER_SIZE = 64
HEADER_FORMAT = "<8s14I"
HEADER_FIELDS = (
    "magic", "crc", "ck_version", "file_version", "zero", "write_mode",
    "hdr1_pack_size", "data_pack_size", "data_unpack_size", "manager_count",
    "object_count", "max_id_saved", "product_version", "product_build",
    "hdr1_unpack_size",
)

CK_CLASS_GROUP = 23
CK_CLASS_DATAARRAY = 52
CK_CLASS_LEVEL = 21
CK_CLASS_3DENTITY = 33
CK_CLASS_3DOBJECT = 41
CK_CLASS_CURVE = 43
CK_CLASS_CURVEPOINT = 36

CK_ARRAYTYPE_INT = 1
CK_ARRAYTYPE_FLOAT = 2
CK_ARRAYTYPE_STRING = 3
CK_ARRAYTYPE_OBJECT = 4

GROUP_IDENT_ALL = 0x000FFFFF
ARRAY_IDENT_FORMAT = 0x00001000
ARRAY_IDENT_DATA = 0x00002000
ARRAY_IDENT_MEMBERS = 0x00004000


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def pad_name(name: str) -> bytes:
    """Virtools object-table names are padded to a 4-byte boundary with NUL."""
    raw = name.encode("utf-8")
    size = (len(raw) + 3) // 4 * 4
    return raw + b"\x00" * (size - len(raw))


@dataclass
class Header:
    magic: bytes
    crc: int
    ck_version: int
    file_version: int
    zero: int
    write_mode: int
    hdr1_pack_size: int
    data_pack_size: int
    data_unpack_size: int
    manager_count: int
    object_count: int
    max_id_saved: int
    product_version: int
    product_build: int
    hdr1_unpack_size: int

    @classmethod
    def parse(cls, blob: bytes) -> "Header":
        if len(blob) < HEADER_SIZE:
            raise ValueError("file is shorter than the NMO header")
        values = struct.unpack_from(HEADER_FORMAT, blob, 0)
        header = cls(*values)
        if header.magic != NEMO_MAGIC:
            raise ValueError(f"not a Nemo file: {header.magic!r}")
        if header.file_version not in (7, 8, 9):
            raise ValueError(f"unsupported NMO file version: {header.file_version}")
        return header

    def pack(self, crc: int | None = None) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            self.magic,
            self.crc if crc is None else crc,
            self.ck_version,
            self.file_version,
            self.zero,
            self.write_mode,
            self.hdr1_pack_size,
            self.data_pack_size,
            self.data_unpack_size,
            self.manager_count,
            self.object_count,
            self.max_id_saved,
            self.product_version,
            self.product_build,
            self.hdr1_unpack_size,
        )


@dataclass
class Identifier:
    ident: int
    pos: int
    next_pos: int
    payload: bytes


@dataclass
class StateChunk:
    data_version: int
    class_id: int
    chunk_version: int
    options: int
    data: bytes
    identifiers: list[Identifier] = field(default_factory=list)

    @classmethod
    def parse(cls, chunk: bytes) -> "StateChunk":
        if len(chunk) < 8:
            raise ValueError("state chunk is too short")
        data_version, class_id, chunk_version, options = chunk[0], chunk[1], chunk[2], chunk[3]
        data_dw_size = u32(chunk, 4)
        data_end = 8 + data_dw_size * 4
        if data_end > len(chunk):
            raise ValueError(
                f"state chunk data size {data_dw_size} exceeds chunk size {len(chunk)}")
        data = chunk[8:data_end]
        state = cls(data_version, class_id, chunk_version, options, data)
        state._parse_identifiers()
        return state

    def _parse_identifiers(self) -> None:
        data = self.data
        data_dw = len(data) // 4
        pos = 0
        seen = set()
        while pos + 1 < data_dw:
            ident = u32(data, pos * 4)
            next_pos = u32(data, (pos + 1) * 4)
            if ident in seen:
                break
            seen.add(ident)
            end = data_dw if next_pos == 0 else next_pos
            start_byte = (pos + 2) * 4
            end_byte = end * 4
            if end_byte < start_byte or end_byte > len(data):
                break
            self.identifiers.append(Identifier(ident, pos, next_pos, data[start_byte:end_byte]))
            if next_pos == 0:
                break
            pos = next_pos

    def find(self, ident: int) -> bytes | None:
        for item in self.identifiers:
            if item.ident == ident:
                return item.payload
        return None


@dataclass
class NmoObject:
    index: int
    object_id: int
    class_id: int
    file_index: int
    name: str
    name_bytes: bytes
    namelen: int
    chunk: bytes
    pack_size: int
    state: StateChunk | None

    @property
    def is_group(self) -> bool:
        return self.class_id == CK_CLASS_GROUP

    @property
    def is_dataarray(self) -> bool:
        return self.class_id == CK_CLASS_DATAARRAY


@dataclass
class ObjectRecord:
    """An object as it will be written to the output object table."""
    object_id: int
    class_id: int
    name: str
    name_bytes: bytes
    namelen: int
    chunk: bytes
    source_index: int | None = None
    source_id: int | None = None

    @classmethod
    def from_object(cls, obj: NmoObject) -> "ObjectRecord":
        return cls(
            object_id=obj.object_id,
            class_id=obj.class_id,
            name=obj.name,
            name_bytes=obj.name_bytes,
            namelen=obj.namelen,
            chunk=obj.chunk,
            source_index=obj.index,
            source_id=obj.object_id,
        )

    @classmethod
    def new(cls, object_id: int, class_id: int, name: str, chunk: bytes) -> "ObjectRecord":
        name_bytes = pad_name(name)
        return cls(
            object_id=object_id,
            class_id=class_id,
            name=name,
            name_bytes=name_bytes,
            namelen=len(name_bytes),
            chunk=chunk,
            source_index=None,
            source_id=None,
        )


class NmoFile:
    def __init__(
        self,
        path: Path,
        blob: bytes,
        header: Header,
        header_raw: bytes,
        header_tail: bytes,
        manager_blob: bytes,
        objects: list[NmoObject],
        data_raw: bytes,
    ) -> None:
        self.path = path
        self.blob = blob
        self.header = header
        self.header_raw = header_raw
        self.header_tail = header_tail
        self.manager_blob = manager_blob
        self.objects = objects
        self.data_raw = data_raw

    @classmethod
    def load(cls, path: str | Path) -> "NmoFile":
        path = Path(path)
        blob = path.read_bytes()
        header = Header.parse(blob)
        header_pack = blob[HEADER_SIZE:HEADER_SIZE + header.hdr1_pack_size]
        data_pack = blob[
            HEADER_SIZE + header.hdr1_pack_size:
            HEADER_SIZE + header.hdr1_pack_size + header.data_pack_size
        ]
        header_raw = zlib.decompress(header_pack) if header.hdr1_pack_size != header.hdr1_unpack_size else header_pack
        data_raw = zlib.decompress(data_pack) if header.data_pack_size != header.data_unpack_size else data_pack
        if len(header_raw) != header.hdr1_unpack_size:
            raise ValueError("unpacked header size mismatch")
        if len(data_raw) != header.data_unpack_size:
            raise ValueError("unpacked data size mismatch")

        objects: list[NmoObject] = []
        pos = 0
        for index in range(header.object_count):
            if pos + 16 > len(header_raw):
                raise ValueError("object table is truncated")
            object_id, class_id, file_index, namelen = struct.unpack_from("<IIII", header_raw, pos)
            pos += 16
            if pos + namelen > len(header_raw):
                raise ValueError("object name is truncated")
            name_bytes = header_raw[pos:pos + namelen]
            pos += namelen
            name = name_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            objects.append(NmoObject(
                index=index,
                object_id=object_id,
                class_id=class_id,
                file_index=file_index,
                name=name,
                name_bytes=name_bytes,
                namelen=namelen,
                chunk=b"",
                pack_size=0,
                state=None,
            ))
        table_end = pos
        header_tail = header_raw[table_end:]

        data_base = HEADER_SIZE + header.hdr1_unpack_size
        first_offset = objects[0].file_index - data_base if objects else 0
        if first_offset < 0 or first_offset > len(data_raw):
            raise ValueError("invalid first object file index")
        manager_blob = data_raw[:first_offset]

        for obj in objects:
            offset = obj.file_index - data_base
            if offset < 0 or offset + 4 > len(data_raw):
                raise ValueError(f"object {obj.name!r} file index is out of range")
            pack_size = u32(data_raw, offset)
            chunk = data_raw[offset + 4:offset + 4 + pack_size]
            if len(chunk) != pack_size:
                raise ValueError(f"object {obj.name!r} chunk is truncated")
            obj.pack_size = pack_size
            obj.chunk = chunk
            try:
                obj.state = StateChunk.parse(chunk)
            except Exception:
                obj.state = None

        return cls(path, blob, header, header_raw, header_tail, manager_blob, objects, data_raw)

    # ----- lookups -----------------------------------------------------
    def object_by_index(self, index: int | None) -> NmoObject | None:
        if index is None or index < 0 or index >= len(self.objects):
            return None
        return self.objects[index]

    def object_by_name(self, name: str, class_id: int | None = None) -> NmoObject | None:
        for obj in self.objects:
            if obj.name != name:
                continue
            if class_id is not None and obj.class_id != class_id:
                continue
            return obj
        return None

    def group_members(self, group_name: str) -> list[NmoObject]:
        group = self.object_by_name(group_name, CK_CLASS_GROUP)
        if group is None or group.state is None:
            return []
        payload = group.state.find(GROUP_IDENT_ALL)
        if payload is None or len(payload) < 4:
            return []
        count = u32(payload, 0)
        if 4 + count * 4 > len(payload):
            return []
        members = []
        for index in struct.unpack_from(f"<{count}I", payload, 4):
            obj = self.object_by_index(index)
            if obj is not None:
                members.append(obj)
        return members

    def dataarray_by_name(self, name: str) -> NmoObject | None:
        return self.object_by_name(name, CK_CLASS_DATAARRAY)

    # ----- writing -----------------------------------------------------
    def build(self, records: Sequence[ObjectRecord]) -> bytes:
        """Serialize a new NMO from a full object record list.

        Existing records keep their name bytes and chunks; new records are
        appended by the caller.  File indices are recomputed from scratch.
        """
        header_entries = bytearray()
        for record in records:
            header_entries += struct.pack("<IIII", record.object_id, record.class_id, 0, record.namelen)
            header_entries += record.name_bytes
        header_size = len(header_entries) + len(self.header_tail)
        data_base = HEADER_SIZE + header_size + len(self.manager_blob)

        file_index = data_base
        header_entries = bytearray()
        data_raw = bytearray(self.manager_blob)
        for record in records:
            header_entries += struct.pack(
                "<IIII", record.object_id, record.class_id, file_index, record.namelen)
            header_entries += record.name_bytes
            chunk_len = len(record.chunk)
            data_raw += struct.pack("<I", chunk_len)
            data_raw += record.chunk
            file_index += 4 + chunk_len
        header_raw = bytes(header_entries) + self.header_tail

        header_pack = zlib.compress(header_raw, 4)
        data_pack = zlib.compress(bytes(data_raw), 4)

        header = Header(
            magic=NEMO_MAGIC,
            crc=0,
            ck_version=self.header.ck_version,
            file_version=self.header.file_version,
            zero=0,
            write_mode=self.header.write_mode,
            hdr1_pack_size=len(header_pack),
            data_pack_size=len(data_pack),
            data_unpack_size=len(data_raw),
            manager_count=self.header.manager_count,
            object_count=len(records),
            max_id_saved=max((r.object_id for r in records), default=0),
            product_version=self.header.product_version,
            product_build=self.header.product_build,
            hdr1_unpack_size=len(header_raw),
        )
        raw_header = header.pack(0)
        crc = zlib.adler32(raw_header, 0)
        crc = zlib.adler32(header_pack, crc)
        crc = zlib.adler32(data_pack, crc)
        return header.pack(crc) + header_pack + data_pack
