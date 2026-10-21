"""CKDataArray state-chunk codec for the FlexGravity editor.

The codec supports the four basic Virtools array cell types used by the
editor: Object, Float, Int and String.  String cells are variable length:
``u32 length`` (including the terminating NUL) followed by the bytes and
4-byte alignment; an empty string is a single zero length.
"""

from __future__ import annotations

import ctypes
import locale
import struct
import sys
from dataclasses import dataclass, field

from .nmo import (
    ARRAY_IDENT_DATA,
    ARRAY_IDENT_FORMAT,
    ARRAY_IDENT_MEMBERS,
    CK_ARRAYTYPE_FLOAT,
    CK_ARRAYTYPE_INT,
    CK_ARRAYTYPE_OBJECT,
    CK_ARRAYTYPE_STRING,
    CK_CLASS_DATAARRAY,
    StateChunk,
    u32,
)

NULL_OBJECT = 0xFFFFFFFF


def system_ansi_encoding() -> str:
    """The code page Virtools 2.1 uses for array strings."""
    if sys.platform == "win32":
        try:
            acp = int(ctypes.windll.kernel32.GetACP())
            if acp:
                return f"cp{acp}"
        except Exception:
            pass
    try:
        return locale.getpreferredencoding(False) or "utf-8"
    except Exception:
        return "utf-8"


DEFAULT_ENCODING = system_ansi_encoding()


@dataclass
class Column:
    name: str
    type: int

    @property
    def is_object(self) -> bool:
        return self.type == CK_ARRAYTYPE_OBJECT

    @property
    def is_float(self) -> bool:
        return self.type == CK_ARRAYTYPE_FLOAT

    @property
    def is_int(self) -> bool:
        return self.type == CK_ARRAYTYPE_INT

    @property
    def is_string(self) -> bool:
        return self.type == CK_ARRAYTYPE_STRING


@dataclass
class ArraySpec:
    name: str
    columns: list[Column]
    rows: list[list[object | None]] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_count(self) -> int:
        return len(self.columns)


def _decode_bytes(raw: bytes, encoding: str | None) -> str:
    for candidate in (encoding, "utf-8", "latin-1"):
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")


def encode_format(columns: list[Column]) -> bytes:
    payload = bytearray(struct.pack("<I", len(columns)))
    for column in columns:
        name_bytes = column.name.encode("utf-8") + b"\x00"
        payload += struct.pack("<I", len(name_bytes))
        payload += name_bytes
        while len(payload) % 4:
            payload += b"\x00"
        payload += struct.pack("<I", column.type)
    return bytes(payload)


def decode_format(payload: bytes) -> list[Column]:
    count = u32(payload, 0)
    pos = 4
    columns: list[Column] = []
    for _ in range(count):
        name_len = u32(payload, pos)
        pos += 4
        raw_name = payload[pos:pos + name_len]
        pos += name_len
        while pos % 4:
            pos += 1
        ctype = u32(payload, pos)
        pos += 4
        name = _decode_bytes(raw_name.split(b"\x00", 1)[0], DEFAULT_ENCODING)
        columns.append(Column(name, ctype))
    if pos != len(payload):
        raise ValueError("array format payload has trailing bytes")
    return columns


def encode_string(text: object | None, encoding: str | None) -> bytes:
    value = "" if text is None else str(text)
    value = value.replace("\x00", "")
    if not value:
        return struct.pack("<I", 0)
    try:
        raw = value.encode(encoding or DEFAULT_ENCODING)
    except (UnicodeEncodeError, LookupError):
        raw = value.encode("utf-8")
    payload = bytearray(struct.pack("<I", len(raw) + 1))
    payload += raw
    payload += b"\x00"
    while len(payload) % 4:
        payload += b"\x00"
    return bytes(payload)


def encode_data(columns: list[Column], rows: list[list[object | None]],
                encoding: str | None = DEFAULT_ENCODING) -> bytes:
    payload = bytearray(struct.pack("<I", len(rows)))
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("row width does not match column count")
        for column, value in zip(columns, row):
            if column.is_object:
                if value is None:
                    payload += struct.pack("<I", NULL_OBJECT)
                else:
                    payload += struct.pack("<I", int(value))
            elif column.is_float:
                payload += struct.pack("<f", 0.0 if value is None else float(value))
            elif column.is_int:
                payload += struct.pack("<i", 0 if value is None else int(value))
            elif column.is_string:
                payload += encode_string(value, encoding)
            else:
                raise ValueError(f"unsupported column type {column.type} for {column.name!r}")
    return bytes(payload)


def _decode_string_at(payload: bytes, pos: int,
                      encoding: str | None) -> tuple[str, int]:
    length = u32(payload, pos)
    pos += 4
    if length == 0:
        value = ""
    else:
        if pos + length > len(payload):
            raise ValueError("string cell exceeds data payload")
        raw = payload[pos:pos + length]
        pos += length
        raw = raw.rstrip(b"\x00")
        value = _decode_bytes(raw, encoding)
    while pos % 4:
        pos += 1
    return value, pos


def decode_data(columns: list[Column], payload: bytes,
                encoding: str | None = DEFAULT_ENCODING) -> list[list[object | None]]:
    row_count = u32(payload, 0)
    pos = 4
    rows: list[list[object | None]] = []
    for _ in range(row_count):
        row: list[object | None] = []
        for column in columns:
            if column.is_object:
                value = u32(payload, pos)
                pos += 4
                row.append(None if value == NULL_OBJECT else value)
            elif column.is_float:
                value = u32(payload, pos)
                pos += 4
                row.append(struct.unpack("<f", struct.pack("<I", value))[0])
            elif column.is_int:
                value = u32(payload, pos)
                pos += 4
                row.append(struct.unpack("<i", struct.pack("<I", value))[0])
            elif column.is_string:
                value, pos = _decode_string_at(payload, pos, encoding)
                row.append(value)
            else:
                raise ValueError(f"unsupported column type {column.type} for {column.name!r}")
        rows.append(row)
    if pos != len(payload):
        raise ValueError("array data payload has trailing bytes")
    return rows


def encode_array(spec: ArraySpec, encoding: str | None = DEFAULT_ENCODING) -> bytes:
    fmt = encode_format(spec.columns)
    data = encode_data(spec.columns, spec.rows, encoding)
    members = struct.pack("<III", 0, 0, NULL_OBJECT)

    payloads = [
        (ARRAY_IDENT_FORMAT, fmt),
        (ARRAY_IDENT_DATA, data),
        (ARRAY_IDENT_MEMBERS, members),
    ]
    entries = []
    pos = 0
    for index, (ident, payload) in enumerate(payloads):
        payload_dw = len(payload) // 4
        next_pos = pos + 2 + payload_dw
        entries.append((ident, next_pos if index + 1 < len(payloads) else 0, payload))
        pos = next_pos
    data_area = bytearray()
    for ident, next_pos, payload in entries:
        data_area += struct.pack("<II", ident, next_pos)
        data_area += payload
    data_dw_size = len(data_area) // 4
    header = bytes([10, CK_CLASS_DATAARRAY, 7, 0x08]) + struct.pack("<I", data_dw_size)
    return header + bytes(data_area)


def decode_array(chunk: bytes, encoding: str | None = DEFAULT_ENCODING) -> ArraySpec:
    state = StateChunk.parse(chunk)
    if state.class_id != CK_CLASS_DATAARRAY:
        raise ValueError(f"not a DataArray chunk (class {state.class_id})")
    fmt = state.find(ARRAY_IDENT_FORMAT)
    data = state.find(ARRAY_IDENT_DATA)
    if fmt is None or data is None:
        raise ValueError("DataArray is missing format or data identifier")
    columns = decode_format(fmt)
    rows = decode_data(columns, data, encoding)
    return ArraySpec("", columns, rows)
