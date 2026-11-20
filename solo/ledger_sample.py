"""Deterministic, really-encoded ledger samples for demos and tests.

The bodies here are produced by the SOLO encoder (or assembled from legal
wire bytes) and then *decoded* by the codec exactly like any other original.
They deliberately include:

* a PAIR character whose second encoded byte is ``0x7C`` (``|``) so that a
  byte-level splitter would manufacture a phantom column;
* a legal no-op mode switch (``FF 00``) inside a line, which emits no
  character and must not count as a separator either;
* cross-page ``=`` inheritance, where the group/worker a page-2 line refers
  to are the *last* values of page 1, not whatever sits next to it in input
  order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .codec import encode_text_bytes
from .ledger import PageVersion

__all__ = ["SampleLedger", "build_sample", "WORKER_WITH_7C"]

# U+4F7C ("佼") encodes as PAIR value 0x017C -> bytes 01 7C in a body, so its
# second byte is exactly the ASCII pipe.  A byte splitter breaks here.
WORKER_WITH_7C = "佼"


@dataclass(frozen=True)
class SampleLedger:
    pages: tuple[PageVersion, ...]
    rotations: dict
    corrected_page1: bytes


def _encode_with_noop_switch(text: str, needle: str, occurrence: int = 1) -> bytes:
    """Encode ``text`` and slip a legal ``FF 00`` no-op switch after ``needle``.

    The inserted switch emits no character, so the decoded text is unchanged;
    only a byte-level reader could be confused by it.
    """

    body = encode_text_bytes(text)
    encoded_needle = encode_text_bytes(needle)
    # Find the occurrence in the *text* and map it to a byte boundary by
    # re-encoding the prefix (the encoder itself is deterministic).
    idx = -1
    start = 0
    for _ in range(occurrence):
        idx = text.find(needle, start)
        if idx < 0:
            raise ValueError(f"needle {needle!r} not found")
        start = idx + len(needle)
    prefix_bytes = len(encode_text_bytes(text[: start]))
    # Insert the no-op switch; both the prefix and suffix are SINGLE here
    # (the needle is ASCII), so the switch is legal and character-neutral.
    return body[:prefix_bytes] + bytes((0xFF, 0x00)) + body[prefix_bytes:]


def build_sample() -> SampleLedger:
    page1 = (
        "PAGE|1|-\n"
        "1|G|A|YES\n"
        "2|=|A|NO\n"
    )
    page2 = (
        "PAGE|2|1\n"
        "3|=|=|MISS\n"
        f"4|H|{WORKER_WITH_7C}|YES\n"
    )
    page3 = (
        "PAGE|3|2\n"
        "5|=|=|NO\n"
    )

    body1 = encode_text_bytes(page1)
    # Put a no-op switch right after the page-1 header line ending.  It is
    # between two ASCII characters, so the decoded text is identical.
    body1 = _encode_with_noop_switch(page1, "\n", 1)
    body2 = encode_text_bytes(page2)
    body3 = encode_text_bytes(page3)

    corrected_page1 = encode_text_bytes(
        "PAGE|1|-\n"
        "1|G|A|MISS\n"  # seq 1 corrected YES -> MISS
        "2|=|A|NO\n"
    )

    rotations = {
        "G": ("A", "B", "C"),
        "H": (WORKER_WITH_7C, "D", "E"),
    }
    return SampleLedger(
        pages=(
            PageVersion(page=1, version=0, body=body1, label="page-1"),
            PageVersion(page=2, version=0, body=body2, label="page-2"),
            PageVersion(page=3, version=0, body=body3, label="page-3"),
        ),
        rotations=rotations,
        corrected_page1=corrected_page1,
    )
