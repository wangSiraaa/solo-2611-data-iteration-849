"""Paged substitute-duty ledger on top of the SOLO codec and revisions.

The shop writes its substitute-call phone log into consecutive SOLO
originals ("pages").  A page starts with a header::

    PAGE|<page number>|<previous page number or ->

and the remaining non-empty lines are notices::

    <sequence>|<group>|<worker>|<response>

``<response>`` is one of the three business codes, written as the encodable
ASCII strings ``YES`` (accepted), ``NO`` (rejected) and ``MISS`` (not
reached).

Field inheritance
-----------------
``=`` in a group or worker field means "inherit the previous valid value":

* on records after the page's first notice line, from the previous notice on
  the same page (``same_page``);
* on a continuation page's first notice line only, from the last notice of
  the page named by the header (``prev_page``).

A start page (``PAGE|n|-``) has no previous page, so ``=`` on its first
notice line is rejected.  If a continuation page's predecessor is missing,
the whole chain is rejected with the affected pages: values are never taken
from some unrelated page that merely happens to be adjacent in the input.

Everything is parsed from *decoded characters* and their byte ranges, never
from raw bytes.  A literal byte ``7C`` inside a PAIR character, or a pure
``FF 00`` / ``FF 01`` mode switch, therefore cannot be mistaken for the
ASCII ``|`` column separator.

Ordering and chains
-------------------
Pages may arrive in any order.  Each page declares at most one predecessor
and therefore at most one successor, and every predecessor chain must reach
a unique start page of its own chain.  Duplicate page/version numbers,
missing predecessors, cycles and two pages claiming the same predecessor
(a "fork") are reported with their affected pages.  Records are interpreted
in page order (that is where inheritance comes from) and only afterwards
emitted as notices sorted by the volume-wide unique sequence number.

Versions and pinned re-queries
------------------------------
A :class:`PageVersion` is an immutable ``(page, version, body)`` triple.
:class:`LedgerBook` keeps one :class:`~solo.revision.RevisionLibrary` per
page, so a correction is a normal text patch and the unmodified original
bytes stay available.  ``book.ledger(select={1: 0})`` pins page 1 to its
first version, making the earlier dispute reproducible after a correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .codec import DecodeError, DecodedDocument, decode_document
from .revision import RevisionLibrary, TextPatch

__all__ = [
    "ByteSpan",
    "ChainLink",
    "FieldFill",
    "ParsedField",
    "ParsedRecord",
    "ParsedPage",
    "PageVersion",
    "Notice",
    "Ledger",
    "LedgerBook",
    "LedgerError",
    "RESPONSE_CODES",
    "parse_page",
    "ledger_from_versions",
    "dispute_replay",
    "format_ledger",
]

RESPONSE_CODES = ("YES", "NO", "MISS")
INHERIT = "="

# Entity kinds used in inheritance chain links.
FIELD_GROUP = "group"
FIELD_WORKER = "worker"


# ---------------------------------------------------------------------------
# Errors


class LedgerError(Exception):
    """A ledger/page/chain problem, with the affected page numbers.

    ``kind`` is a stable machine-readable code: ``duplicate_page``,
    ``missing_predecessor``, ``fork``, ``cycle``, ``bad_header``,
    ``bad_record``, ``bad_response``, ``duplicate_sequence``,
    ``inherit_without_source``, ``page_mismatch``, ``unknown_version`` or
    ``duplicate_page_version``.
    """

    def __init__(self, kind: str, message: str, pages: Iterable[int] = ()) -> None:
        self.kind = kind
        self.pages = tuple(sorted(set(int(p) for p in pages)))
        self.message = message
        where = f" (pages {list(self.pages)})" if self.pages else ""
        super().__init__(f"{kind}: {message}{where}")


# ---------------------------------------------------------------------------
# Value types


@dataclass(frozen=True)
class ByteSpan:
    """Half-open byte interval inside one exact page-version body."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid byte span [{self.start},{self.end})")

    def as_tuple(self) -> tuple[int, int]:
        return (self.start, self.end)

    def __str__(self) -> str:
        return f"[{self.start},{self.end})"


@dataclass(frozen=True)
class ParsedField:
    text: str
    span: ByteSpan


@dataclass(frozen=True)
class ParsedRecord:
    line_no: int
    seq: int
    seq_field: ParsedField
    group: ParsedField
    worker: ParsedField
    response: ParsedField
    line_text: str = ""

    @property
    def response_code(self) -> str:
        return self.response.text


@dataclass(frozen=True)
class ParsedPage:
    page: int
    prev: int | None
    version: int
    body: bytes
    text: str
    records: tuple[ParsedRecord, ...]
    header_fields: tuple[ParsedField, ...]


@dataclass(frozen=True)
class ChainLink:
    """One hop of a fill chain, from the ultimate explicit value onward."""

    page: int
    version: int
    seq: int
    field: str
    value: str
    source: str  # "explicit" | "same_page" | "prev_page"
    token_span: ByteSpan

    def __str__(self) -> str:
        return (
            f"p{self.page}v{self.version}:seq{self.seq}.{self.field}"
            f"={self.value!r}[{self.source}]{self.token_span}"
        )


@dataclass(frozen=True)
class FieldFill:
    """Resolved value of one field plus where it actually came from."""

    value: str
    source: str  # "explicit" | "same_page" | "prev_page"
    token_span: ByteSpan  # the bytes of this field on this page
    origin_page: int
    origin_version: int
    origin_seq: int
    origin_span: ByteSpan  # bytes of the ultimate explicit value
    chain: tuple[ChainLink, ...]

    @property
    def inherited(self) -> bool:
        return self.source != "explicit"


@dataclass(frozen=True)
class PageVersion:
    """A frozen body of one page version."""

    page: int
    version: int
    body: bytes
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.body, (bytes, bytearray)):
            raise TypeError("PageVersion.body must be bytes")
        object.__setattr__(self, "body", bytes(self.body))


@dataclass(frozen=True)
class Notice:
    """One emitted notification, after inheritance and sequence sorting."""

    seq: int
    group: str
    worker: str
    response: str
    page: int
    version: int
    line_no: int
    seq_fill: FieldFill
    group_fill: FieldFill
    worker_fill: FieldFill
    response_fill: FieldFill
    raw_line: str

    @property
    def time(self) -> int:
        """The sequence number doubles as the notification time."""

        return self.seq

    def field_fill(self, name: str) -> FieldFill:
        try:
            return {
                "seq": self.seq_fill,
                "group": self.group_fill,
                "worker": self.worker_fill,
                "response": self.response_fill,
            }[name]
        except KeyError:
            raise KeyError(f"unknown notice field {name!r}") from None

    def row(self) -> tuple[int, str, str, str]:
        return (self.seq, self.group, self.worker, self.response)


@dataclass(frozen=True)
class Ledger:
    """Notices plus their pages, chains and per-field origins."""

    notices: tuple[Notice, ...]
    pages: tuple[ParsedPage, ...]
    chains: tuple[tuple[int, ...], ...]
    _by_seq: Mapping[int, Notice] = field(default_factory=dict, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.notices)

    def by_seq(self, seq: int) -> Notice:
        return self._by_seq[seq]

    def page(self, page: int) -> ParsedPage:
        for p in self.pages:
            if p.page == page:
                return p
        raise KeyError(page)

    def rows(self) -> list[tuple[int, str, str, str]]:
        return [n.row() for n in self.notices]

    def rows_in_page_order(self) -> list[tuple[int, str, str, str]]:
        rows: list[tuple[int, str, str, str]] = []
        for page in self.pages:
            for rec in page.records:
                seq = rec.seq
                notice = self._by_seq[seq]
                rows.append(notice.row())
        return rows


# ---------------------------------------------------------------------------
# Parsing


def _char_boundary(doc: DecodedDocument, index: int) -> int:
    n = len(doc.text)
    if doc.starts is None:
        raise ValueError("decoded document lacks byte ranges")
    if index < n:
        return int(doc.starts[index])
    return int(doc.ends[n - 1]) if n else 0


def _byte_span(doc: DecodedDocument, start_char: int, end_char: int) -> ByteSpan:
    """Byte interval covering decoded characters ``[start_char, end_char)``.

    A pure mode switch between two characters lies outside both character
    ranges; for a non-empty field the span from the first character start to
    the last character end therefore still covers any switches between the
    field's characters.  Empty fields collapse to a character boundary.
    """

    if doc.starts is None or doc.ends is None:
        raise ValueError("decoded document lacks byte ranges")
    if end_char > start_char:
        return ByteSpan(int(doc.starts[start_char]), int(doc.ends[end_char - 1]))
    pos = _char_boundary(doc, start_char)
    return ByteSpan(pos, pos)


def _split_fields(
    doc: DecodedDocument, start_char: int, end_char: int
) -> list[ParsedField]:
    """Split ``doc.text[start_char:end_char]`` on decoded ASCII ``|``.

    Splitting decoded *characters* means a byte 0x7C that happens to be the
    low byte of a PAIR character is data, not a separator.
    """

    text = doc.text
    fields: list[ParsedField] = []
    field_start = start_char
    i = start_char
    while i < end_char:
        if text[i] == "|":
            fields.append(
                ParsedField(text[field_start:i], _byte_span(doc, field_start, i))
            )
            field_start = i + 1
        i += 1
    fields.append(ParsedField(text[field_start:end_char], _byte_span(doc, field_start, end_char)))
    return fields


def _emit_lines(doc: DecodedDocument) -> list[tuple[int, int, int, int]]:
    """Return ``(line_no, start_char, end_char, newline_char)`` per raw line."""

    lines: list[tuple[int, int, int, int]] = []
    text = doc.text
    n = len(text)
    start = 0
    line_no = 0
    for i in range(n):
        if text[i] == "\n":
            lines.append((line_no, start, i, i))
            start = i + 1
            line_no += 1
    if start < n:
        lines.append((line_no, start, n, n))
    return lines


def _parse_int_field(field: ParsedField, what: str, line_no: int) -> int:
    raw = field.text.strip()
    try:
        value = int(raw, 10)
    except ValueError:
        raise LedgerError(
            "bad_record",
            f"line {line_no + 1}: {what} {field.text!r} is not an integer",
        ) from None
    if value < 0:
        raise LedgerError("bad_record", f"line {line_no + 1}: {what} {field.text!r} is negative")
    return value


def parse_page(
    body: bytes | bytearray | memoryview,
    *,
    page: int | None = None,
    version: int = 0,
    chunk_size: int = 131072,
) -> ParsedPage:
    """Decode and parse one page body.

    ``page`` overrides the page number from the header (an external registry
    may know the page identity even if the header were revised).  If it is
    given it must agree with the header.
    """

    raw = bytes(body)
    try:
        doc = decode_document(raw, chunk_size=chunk_size)
    except DecodeError as exc:
        raise LedgerError(
            "bad_record", f"page body does not decode: {exc}", (page,) if page is not None else ()
        ) from exc
    return _parse_decoded(doc, raw, page=page, version=version)


def _parse_decoded(
    doc: DecodedDocument,
    raw: bytes,
    *,
    page: int | None,
    version: int,
) -> ParsedPage:
    lines = _emit_lines(doc)
    if not lines:
        raise LedgerError("bad_header", "page has no header line", (page,) if page is not None else ())

    header_line, hs, he, _ = lines[0]
    header = _split_fields(doc, hs, he)
    if len(header) != 3 or header[0].text != "PAGE":
        raise LedgerError(
            "bad_header",
            f"line {header_line + 1}: expected 'PAGE|<page>|<prev or ->', got {doc.text[hs:he]!r}",
            (page,) if page is not None else (),
        )
    header_page = _parse_int_field(header[1], "page number", header_line)
    if page is not None and header_page != page:
        raise LedgerError(
            "page_mismatch",
            f"header says page {header_page}, registry says page {page}",
            (page, header_page),
        )
    page_no = header_page
    prev_text = header[2].text.strip()
    if prev_text in ("-", ""):
        prev: int | None = None
    else:
        try:
            prev = int(prev_text, 10)
        except ValueError:
            raise LedgerError(
                "bad_header",
                f"line {header_line + 1}: previous page {header[2].text!r} is not an integer",
                (page_no,),
            ) from None
        if prev < 0:
            raise LedgerError("bad_header", f"line {header_line + 1}: negative previous page", (page_no,))
    if prev == page_no:
        raise LedgerError("bad_header", f"page {page_no} names itself as its predecessor", (page_no,))

    records: list[ParsedRecord] = []
    seen_seq: set[int] = set()
    for line_no, start, end, _nl in lines[1:]:
        if not doc.text[start:end].strip():
            continue
        fields = _split_fields(doc, start, end)
        if len(fields) != 4:
            raise LedgerError(
                "bad_record",
                f"line {line_no + 1}: expected 4 fields, got {len(fields)} in "
                f"{doc.text[start:end]!r}",
                (page_no,),
            )
        seq_field, group, worker, response = fields
        seq = _parse_int_field(seq_field, "sequence", line_no)
        if seq in seen_seq:
            raise LedgerError(
                "duplicate_sequence", f"line {line_no + 1}: sequence {seq} repeats on the page", (page_no,)
            )
        seen_seq.add(seq)
        for name, f in (("group", group), ("worker", worker)):
            if f.text != INHERIT and INHERIT in f.text:
                raise LedgerError(
                    "bad_record",
                    f"line {line_no + 1}: {name} {f.text!r} mixes '=' with other text",
                    (page_no,),
                )
        if response.text not in RESPONSE_CODES:
            # Also rejects '=' in the response column: business codes are
            # never inherited.
            raise LedgerError(
                "bad_response",
                f"line {line_no + 1}: response {response.text!r} is not one of "
                f"{RESPONSE_CODES}",
                (page_no,),
            )
        records.append(
            ParsedRecord(
                line_no=line_no,
                seq=seq,
                seq_field=seq_field,
                group=group,
                worker=worker,
                response=response,
                line_text=doc.text[start:end],
            )
        )
    return ParsedPage(
        page=page_no,
        prev=prev,
        version=version,
        body=raw,
        text=doc.text,
        records=tuple(records),
        header_fields=tuple(header),
    )


# ---------------------------------------------------------------------------
# Chain resolution


def _check_predecessors(pages: Sequence[ParsedPage]) -> None:
    by_page: dict[int, ParsedPage] = {}
    for p in pages:
        if p.page in by_page:
            raise LedgerError("duplicate_page", f"page {p.page} appears more than once", (p.page,))
        by_page[p.page] = p

    # A page naming a predecessor means that predecessor must be present.  A
    # page may be in the middle of a longer chain, but it may have at most one
    # successor; two pages claiming the same predecessor is a fork.
    successors: dict[int, list[int]] = {}
    for p in pages:
        if p.prev is None:
            continue
        if p.prev not in by_page:
            raise LedgerError(
                "missing_predecessor",
                f"page {p.page} continues missing page {p.prev}",
                (p.page, p.prev),
            )
        successors.setdefault(p.prev, []).append(p.page)
    for pred_page, succ in successors.items():
        if len(succ) > 1:
            raise LedgerError(
                "fork",
                f"page {pred_page} has two successors {sorted(succ)}",
                (pred_page, *succ),
            )

    # Every page must trace back to a start page; otherwise there is a cycle.
    for p in pages:
        seen: set[int] = set()
        cur: int | None = p.page
        while cur is not None:
            if cur in seen:
                cyc = sorted(seen)
                raise LedgerError("cycle", f"predecessor cycle among pages {cyc}", cyc)
            seen.add(cur)
            cur = by_page[cur].prev


def _chain_order(pages: Sequence[ParsedPage]) -> list[list[ParsedPage]]:
    """Group pages into ordered chains (start page first)."""

    by_page = {p.page: p for p in pages}
    successor: dict[int, int] = {}
    starts: list[int] = []
    for p in pages:
        if p.prev is None:
            starts.append(p.page)
        else:
            successor[p.prev] = p.page
    starts.sort()
    chains: list[list[ParsedPage]] = []
    for start in starts:
        chain: list[ParsedPage] = []
        cur: int | None = start
        while cur is not None:
            chain.append(by_page[cur])
            cur = successor.get(cur)
        chains.append(chain)
    return chains


# ---------------------------------------------------------------------------
# Ledger assembly


def _resolve_field(
    *,
    field: str,
    token: ParsedField,
    prev_explicit: FieldFill | None,
    page: ParsedPage,
    seq: int,
    line_no: int,
    prev_page_last: FieldFill | None,
    prev_page_no: int | None,
    first_on_page: bool,
) -> FieldFill:
    if token.text != INHERIT:
        origin = ChainLink(
            page=page.page,
            version=page.version,
            seq=seq,
            field=field,
            value=token.text,
            source="explicit",
            token_span=token.span,
        )
        return FieldFill(
            value=token.text,
            source="explicit",
            token_span=token.span,
            origin_page=page.page,
            origin_version=page.version,
            origin_seq=seq,
            origin_span=token.span,
            chain=(origin,),
        )

    if first_on_page:
        if prev_page_no is None:
            raise LedgerError(
                "inherit_without_source",
                f"page {page.page} line {line_no + 1}: '=' in {field} but the page "
                f"starts its own chain (no previous page)",
                (page.page,),
            )
        source = "prev_page"
        base = prev_page_last
        if base is None:
            raise LedgerError(
                "inherit_without_source",
                f"page {page.page} line {line_no + 1}: '=' in {field} but page "
                f"{prev_page_no} has no valid value to inherit",
                (prev_page_no, page.page),
            )
    else:
        source = "same_page"
        if prev_explicit is None:
            raise LedgerError(
                "inherit_without_source",
                f"page {page.page} line {line_no + 1}: '=' in {field} with no "
                f"preceding valid value on the page",
                (page.page,),
            )
        base = prev_explicit

    hop = ChainLink(
        page=page.page,
        version=page.version,
        seq=seq,
        field=field,
        value=base.value,
        source=source,
        token_span=token.span,
    )
    return FieldFill(
        value=base.value,
        source=source,
        token_span=token.span,
        origin_page=base.origin_page,
        origin_version=base.origin_version,
        origin_seq=base.origin_seq,
        origin_span=base.origin_span,
        chain=base.chain + (hop,),
    )


def ledger_from_versions(
    versions: Sequence[PageVersion],
    *,
    chunk_size: int = 131072,
) -> Ledger:
    """Parse pinned page versions into a fully resolved :class:`Ledger`."""

    if not versions:
        raise LedgerError("bad_repository", "no page versions supplied")

    by_page: dict[int, PageVersion] = {}
    for item in versions:
        key = item.page
        existing = by_page.get(key)
        if existing is not None:
            if existing.version == item.version:
                raise LedgerError(
                    "duplicate_page_version",
                    f"page {key} version {item.version} supplied twice",
                    (key,),
                )
            raise LedgerError(
                "duplicate_page",
                f"page {key} supplied with several versions "
                f"({existing.version}, {item.version}); pin one",
                (key,),
            )
        by_page[key] = item

    parsed: list[ParsedPage] = []
    for item in sorted(versions, key=lambda v: (v.page, v.version)):
        parsed.append(
            parse_page(
                item.body, page=item.page, version=item.version, chunk_size=chunk_size
            )
        )
    _check_predecessors(parsed)
    chains = _chain_order(parsed)

    notices: list[Notice] = []
    seen_seq: dict[int, int] = {}
    for chain in chains:
        # Last resolved value per field carried across the page boundary.
        prev_page_no: int | None = None
        prev_group: FieldFill | None = None
        prev_worker: FieldFill | None = None
        for index, page in enumerate(chain):
            last_group: FieldFill | None = None
            last_worker: FieldFill | None = None
            for rec_index, rec in enumerate(page.records):
                first_on_page = rec_index == 0
                group_fill = _resolve_field(
                    field=FIELD_GROUP,
                    token=rec.group,
                    prev_explicit=last_group,
                    page=page,
                    seq=rec.seq,
                    line_no=rec.line_no,
                    prev_page_last=prev_group,
                    prev_page_no=prev_page_no,
                    first_on_page=first_on_page,
                )
                worker_fill = _resolve_field(
                    field=FIELD_WORKER,
                    token=rec.worker,
                    prev_explicit=last_worker,
                    page=page,
                    seq=rec.seq,
                    line_no=rec.line_no,
                    prev_page_last=prev_worker,
                    prev_page_no=prev_page_no,
                    first_on_page=first_on_page,
                )
                # The previously valid value on this page.  An inherited fill
                # already carries the true origin and value, so it is a valid
                # source for the next record too.
                last_group = group_fill
                last_worker = worker_fill

                if rec.seq in seen_seq:
                    raise LedgerError(
                        "duplicate_sequence",
                        f"sequence {rec.seq} appears on pages {seen_seq[rec.seq]} and {page.page}",
                        (seen_seq[rec.seq], page.page),
                    )
                seen_seq[rec.seq] = page.page

                seq_fill = FieldFill(
                    value=str(rec.seq),
                    source="explicit",
                    token_span=rec.seq_field.span,
                    origin_page=page.page,
                    origin_version=page.version,
                    origin_seq=rec.seq,
                    origin_span=rec.seq_field.span,
                    chain=(
                        ChainLink(
                            page=page.page,
                            version=page.version,
                            seq=rec.seq,
                            field="seq",
                            value=str(rec.seq),
                            source="explicit",
                            token_span=rec.seq_field.span,
                        ),
                    ),
                )
                response_fill = FieldFill(
                    value=rec.response.text,
                    source="explicit",
                    token_span=rec.response.span,
                    origin_page=page.page,
                    origin_version=page.version,
                    origin_seq=rec.seq,
                    origin_span=rec.response.span,
                    chain=(
                        ChainLink(
                            page=page.page,
                            version=page.version,
                            seq=rec.seq,
                            field="response",
                            value=rec.response.text,
                            source="explicit",
                            token_span=rec.response.span,
                        ),
                    ),
                )
                raw_line = rec.line_text
                notices.append(
                    Notice(
                        seq=rec.seq,
                        group=group_fill.value,
                        worker=worker_fill.value,
                        response=rec.response.text,
                        page=page.page,
                        version=page.version,
                        line_no=rec.line_no,
                        seq_fill=seq_fill,
                        group_fill=group_fill,
                        worker_fill=worker_fill,
                        response_fill=response_fill,
                        raw_line=raw_line,
                    )
                )
            prev_page_no = page.page
            # Only the designated predecessor's own last resolved values may
            # feed the next page's first line.  If this page has no records,
            # these become None so the next page fails rather than reaching
            # further back.
            prev_group = last_group
            prev_worker = last_worker

    notices.sort(key=lambda n: n.seq)
    by_seq = {n.seq: n for n in notices}
    return Ledger(
        notices=tuple(notices),
        pages=tuple(p for chain in chains for p in chain),
        chains=tuple(tuple(p.page for p in chain) for chain in chains),
        _by_seq=by_seq,
    )


# ---------------------------------------------------------------------------
# Book: one revision library per page


class LedgerBook:
    """Per-page revision libraries with frozen, pinnable versions.

    Each page keeps its own :class:`~solo.revision.RevisionLibrary`; the
    initial body is version 0 and corrections are ordinary text patches on a
    pinned base.  Old versions remain available, so a dispute can be replayed
    against the exact bytes it was decided on.
    """

    def __init__(self, *, chunk_size: int = 131072) -> None:
        self._chunk_size = chunk_size
        self._libs: dict[int, RevisionLibrary] = {}

    def add_page(
        self, page: int, body: bytes | bytearray | memoryview
    ):
        if page in self._libs:
            raise LedgerError("duplicate_page", f"page {page} already added", (page,))
        lib = RevisionLibrary()
        version = lib.add_original(body, chunk_size=self._chunk_size)
        # Decode-time errors surface here as DecodeError; convert to LedgerError
        # so callers see page context.
        try:
            parse_page(version.body, page=page, version=0, chunk_size=self._chunk_size)
        except LedgerError:
            raise
        self._libs[page] = lib
        return version

    def pages(self) -> list[int]:
        return sorted(self._libs)

    def version(self, page: int, version: int):
        lib = self._lib(page)
        return lib.get(version)

    def latest(self, page: int):
        lib = self._lib(page)
        return lib.get(max(v.version_id for v in lib.versions()))

    def versions_for(self, page: int):
        return self._lib(page).versions()

    def apply(
        self, page: int, base_version: int, patches: Sequence[TextPatch]
    ):
        """Apply a correction to one pinned page version."""

        return self._lib(page).apply(base_version, patches, chunk_size=self._chunk_size)

    def _lib(self, page: int) -> RevisionLibrary:
        try:
            return self._libs[page]
        except KeyError:
            raise LedgerError("unknown_version", f"page {page} was never added", (page,)) from None

    def ledger(self, select: Mapping[int, int] | None = None) -> Ledger:
        """Build a ledger from the selected (default: latest) version per page.

        ``select`` pins individual pages to a specific version; every other
        page uses its latest version.  This is how a corrected ledger and the
        original disputed ledger are compared side by side.
        """

        pins = dict(select or {})
        versions: list[PageVersion] = []
        for page in self.pages():
            if page in pins:
                ver = self.version(page, pins[page])
            else:
                ver = self.latest(page)
            versions.append(
                PageVersion(page=page, version=ver.version_id, body=ver.body)
            )
        return ledger_from_versions(versions, chunk_size=self._chunk_size)


# ---------------------------------------------------------------------------
# Replaying a dispute with two frozen originals


def dispute_replay(
    pages_before: Sequence[PageVersion],
    pages_after: Sequence[PageVersion],
    roster,
    *,
    chunk_size: int = 131072,
) -> dict:
    """Replay the same dispute from a frozen old ledger and a corrected one.

    ``pages_before`` and ``pages_after`` each hold one version per page (the
    old and corrected originals).  The union rotation is unchanged between the
    two runs; only the ledger bytes differ.  Returns both ledgers, both call
    lists, the disputes that changed, and the queue orders afterwards.
    """

    old = ledger_from_versions(pages_before, chunk_size=chunk_size)
    new = ledger_from_versions(pages_after, chunk_size=chunk_size)
    old_calls = roster.replay(old.notices)
    new_calls = roster.replay(new.notices)

    old_by_seq = {c.seq: c for c in old_calls}
    new_by_seq = {c.seq: c for c in new_calls}
    changed: list[dict] = []
    for seq in sorted(set(old_by_seq) | set(new_by_seq)):
        a = old_by_seq.get(seq)
        b = new_by_seq.get(seq)
        if a is None or b is None:
            changed.append(
                {"seq": seq, "old": a, "new": b, "reason": "seq appears in only one version"}
            )
            continue
        if (a.actual, a.expected, a.disputed, a.response, a.order_after) != (
            b.actual,
            b.expected,
            b.disputed,
            b.response,
            b.order_after,
        ):
            changed.append({"seq": seq, "old": a, "new": b, "reason": "outcome changed"})
    return {
        "old_ledger": old,
        "new_ledger": new,
        "old_calls": old_calls,
        "new_calls": new_calls,
        "changed": changed,
        "old_order": roster.order_after(old.notices),
        "new_order": roster.order_after(new.notices),
    }


def format_ledger(
    ledger: Ledger,
    *,
    calls: Sequence | None = None,
    changed_seqs: Iterable[int] = (),
) -> list[str]:
    """Human-readable notification detail: values, source, chain, byte range."""

    changed = set(changed_seqs)
    calls_by_seq = {c.seq: c for c in calls} if calls else {}
    out: list[str] = []
    for n in ledger.notices:
        mark = "*" if n.seq in changed else " "
        out.append(
            f"{mark} {n.seq}|{n.group}|{n.worker}|{n.response}  "
            f"page {n.page} v{n.version} line {n.line_no + 1}"
        )
        for name in ("seq", "group", "worker", "response"):
            f = n.field_fill(name)
            where = (
                f"origin p{f.origin_page}v{f.origin_version} seq {f.origin_seq} "
                f"{f.origin_span}"
            )
            if f.inherited:
                chain = " -> ".join(str(link) for link in f.chain)
                out.append(
                    f"    {name}: {f.value!r} from {f.source} {f.token_span}; "
                    f"{where}; chain {chain}"
                )
            else:
                out.append(
                    f"    {name}: {f.value!r} explicit {f.token_span}; {where}"
                )
        call = calls_by_seq.get(n.seq)
        if call is not None:
            verdict = "DISPUTE" if call.disputed else "ok"
            out.append(
                f"    union: expected {call.expected!r} actual {call.actual!r} "
                f"[{verdict}] order_before {call.order_before} -> after "
                f"{call.order_after}"
            )
            if call.skipped:
                out.append(
                    "    skipped: "
                    + ", ".join(f"{w} ({why})" for w, why in call.skipped)
                )
    return out
