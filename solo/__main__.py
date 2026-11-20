"""Command line entry points.

    python3 -m solo decode FILE [--chunk N] [--checkpoint OFFSET]
    python3 -m solo encode TEXT
    python3 -m solo revise ORIGINAL PATCHES.json
    python3 -m solo demo
    python3 -m solo check FILE
    python3 -m solo ledger CONFIG.json [--time T] [--group G]

``revise`` reads a JSON batch:

    {"patches": [
        {"replace": [start_char, end_char, "text"]},
        {"preserve": [start_byte, end_byte]}
    ]}

``ledger`` reads a JSON manifest of page files (each a real SOLO body) plus an
optional union roster and prints notices, inheritance chains, byte ranges and
the expected caller.  It never rewrites a page; corrections in the manifest
are separate pinned versions::

    {
      "pages": [{"page": 1, "version": 0, "file": "p1.bin"}, ...],
      "roster": {"G": {"rotation": ["A", "B", "C"],
                        "qualified": ["A", "B", "C"],
                        "qualified_from": {},
                        "leaves": [{"worker": "B", "start": 1, "end": 2}]}}
    }
"""

from __future__ import annotations

import argparse
import json
import sys

from .codec import CheckpointError, DecodeError, EncodeError, decode_document, encode_text
from .ledger import (
    LedgerError,
    PageVersion,
    format_ledger,
    ledger_from_versions,
)
from .revision import RevisionError, RevisionLibrary, preserve, replace
from .roster import GroupRoster, Leave, Roster, RosterError


def _load(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def cmd_decode(args: argparse.Namespace) -> int:
    body = _load(args.file)
    try:
        if args.checkpoint is not None:
            lib = RevisionLibrary()
            version = lib.add_original(body)
            cp = version.checkpoint_at(min(args.checkpoint, len(body)))
            dec = cp.resume(body)
            chars = dec.feed(body[cp.position :])
            final_mode = dec.finish()
            print("".join(c.char for c in chars))
            print(f"# resumed at {cp.position}, final mode {final_mode}", file=sys.stderr)
        else:
            doc = decode_document(body, chunk_size=args.chunk)
            print(doc.text)
            print(
                f"# {len(doc.text)} chars, final mode {doc.final_mode}", file=sys.stderr
            )
    except DecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except CheckpointError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def cmd_encode(args: argparse.Namespace) -> int:
    try:
        encoded = encode_text(args.text)
    except EncodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(encoded.data)
    return 0


def cmd_revise(args: argparse.Namespace) -> int:
    batch = json.loads(_load(args.patches).decode("utf-8"))
    patches = []
    for item in batch["patches"]:
        if "replace" in item:
            start, end, text = item["replace"]
            patches.append(replace(start, end, text))
        elif "preserve" in item:
            start, end = item["preserve"]
            patches.append(preserve(start, end))
        else:
            print(f"unknown patch {item}", file=sys.stderr)
            return 2
    lib = RevisionLibrary()
    try:
        original = lib.add_original(_load(args.original))
        version = lib.apply(0, patches)
    except (DecodeError, RevisionError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(version.body)
    print(f"# {original.summary()}", file=sys.stderr)
    print(f"# {version.summary()}", file=sys.stderr)
    for rec in version.records:
        print(f"# {rec.summary()}", file=sys.stderr)
    for adj in version.boundary_adjustments:
        print(f"# boundary {adj.summary()}", file=sys.stderr)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import demo

    demo()
    return 0


def _load_roster(spec: dict) -> Roster:
    groups = []
    for name, cfg in spec.items():
        rotation = tuple(cfg["rotation"])
        leaves = tuple(
            Leave(item["worker"], int(item["start"]), int(item["end"]))
            for item in cfg.get("leaves", [])
        )
        groups.append(
            GroupRoster(
                group=name,
                rotation=rotation,
                qualified=frozenset(cfg.get("qualified", rotation)),
                qualified_from={
                    worker: int(since)
                    for worker, since in cfg.get("qualified_from", {}).items()
                },
                leaves=leaves,
            )
        )
    return Roster(groups)


def cmd_ledger(args: argparse.Namespace) -> int:
    config = json.loads(_load(args.config).decode("utf-8"))
    versions = [
        PageVersion(
            page=int(item["page"]),
            version=int(item.get("version", 0)),
            body=_load(item["file"]),
            label=item.get("label", ""),
        )
        for item in config["pages"]
    ]
    try:
        ledger = ledger_from_versions(versions)
    except LedgerError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    calls = None
    if "roster" in config:
        try:
            roster = _load_roster(config["roster"])
        except (RosterError, KeyError, TypeError) as exc:
            print(f"roster error: {exc}", file=sys.stderr)
            return 2
        selected = [
            n
            for n in ledger.notices
            if (args.group is None or n.group == args.group)
            and (args.time is None or n.time <= args.time)
        ]
        try:
            calls = roster.replay(selected)
        except RosterError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.time is not None and args.group is not None:
            upcoming = roster.who_at(
                [n for n in ledger.notices if n.group == args.group], args.time, args.group
            )
            print(f"# expected caller at time {args.time}: {upcoming[args.group]}", file=sys.stderr)

    for line in format_ledger(ledger, calls=calls):
        print(line)
    if calls is not None:
        disputes = [c for c in calls if c.disputed]
        print(f"# {len(calls)} notices, {len(disputes)} disputes", file=sys.stderr)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    body = _load(args.file)
    try:
        doc = decode_document(body)
    except DecodeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"ok chars={len(doc.text)} bytes={len(body)} final={doc.final_mode}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("decode", help="decode a SOLO body")
    p.add_argument("file")
    p.add_argument("--chunk", type=int, default=131072)
    p.add_argument("--checkpoint", type=int, default=None)
    p.set_defaults(func=cmd_decode)

    p = sub.add_parser("encode", help="encode representable text")
    p.add_argument("text")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("revise", help="apply a JSON patch batch")
    p.add_argument("original")
    p.add_argument("patches")
    p.set_defaults(func=cmd_revise)

    p = sub.add_parser("demo", help="write a demo and artifacts")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("check", help="validate a body")
    p.add_argument("file")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("ledger", help="resolve a paged ledger and union roster")
    p.add_argument("config")
    p.add_argument("--time", type=int, default=None)
    p.add_argument("--group", default=None)
    p.set_defaults(func=cmd_ledger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
