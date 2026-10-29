"""ANSI/GBK text helpers for Virtools 2.1 names and strings."""

from __future__ import annotations

import ctypes
import locale
import sys


def system_ansi_encoding() -> str:
    """The code page Virtools 2.1 uses for object names and array strings."""
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


def encode_ansi(text: str, encoding: str | None = None, *, strict: bool = False) -> bytes:
    """Encode text for VT 2.1.

    ``strict`` is used for object names: a name that cannot be represented in
    the ANSI code page is an export error, because the mod matches object names
    against GBK array strings.
    """
    value = "" if text is None else str(text)
    candidates = [encoding or DEFAULT_ENCODING]
    if not strict:
        candidates.extend(["utf-8", "latin-1"])
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return value.encode(candidate)
        except (UnicodeEncodeError, LookupError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return value.encode("latin-1")


def decode_ansi(raw: bytes, encoding: str | None = None) -> str:
    """Decode a VT 2.1 name/string: ANSI first, then UTF-8, then Latin-1."""
    for candidate in (encoding or DEFAULT_ENCODING, "utf-8", "latin-1"):
        if not candidate:
            continue
        try:
            return raw.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors="replace")
