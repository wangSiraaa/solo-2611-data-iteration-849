"""Tests for the paged ledger and the union duty roster.

Run with:  python3 -m unittest discover -s tests -v

The tests use *real* SOLO bodies produced by the encoder (never hand-written
strangeled text) so that byte-level traps -- a ``7C`` low byte inside a PAIR
character, a no-op ``FF 00`` switch -- are genuinely exercised.  A small,
deliberately naive hand-written ledger is kept beside the implementation as an
independent arithmetic check for every notification row.
"""

from __future__ import annotations

import unittest

from solo.codec import decode_document, encode_text_bytes
from solo.ledger import (
    LedgerError,
    PageVersion,
    dispute_replay,
    format_ledger,
    ledger_from_versions,
    parse_page,
)
from solo.ledger_sample import WORKER_WITH_7C, build_sample
from solo.roster import GroupRoster, Leave, Roster, RosterError


def pv(page: int, text: str, version: int = 0) -> PageVersion:
    return PageVersion(page=page, version=version, body=encode_text_bytes(text))


def pages(*items: PageVersion) -> tuple[PageVersion, ...]:
    return tuple(items)


def ledger_of(*items: PageVersion):
    return ledger_from_versions(items)


def roster_of(rotations, **kwargs):
    return Roster(
        [
            GroupRoster(
                group=g,
                rotation=tuple(rot),
                qualified=frozenset(kwargs.get("qualified", {}).get(g, rot)),
                qualified_from=kwargs.get("qualified_from", {}).get(g, {}),
                leaves=tuple(kwargs.get("leaves", {}).get(g, ())),
            )
            for g, rot in rotations.items()
        ]
    )


# ---------------------------------------------------------------------------
# A tiny independent arithmetic ledger: exactly the statement's example.

# seq -> (group, worker, response), written by hand from the sample pages.
EXAMPLE_PAGE_1 = "PAGE|1|-\n1|G|A|YES\n2|=|A|NO\n"
EXAMPLE_PAGE_2 = "PAGE|2|1\n3|=|=|MISS\n"
HAND_ROWS = [
    (1, "G", "A", "YES"),
    (2, "G", "A", "NO"),
    (3, "G", "A", "MISS"),
]


class LedgerParseTests(unittest.TestCase):
    def test_rows_match_hand_ledger(self):
        led = ledger_of(pv(1, EXAMPLE_PAGE_1), pv(2, EXAMPLE_PAGE_2))
        self.assertEqual(led.rows(), HAND_ROWS)

    def test_page_order_differs_from_seq_order(self):
        # Page 2 arrives first, but its continuation depends on page 1.
        led = ledger_of(pv(2, EXAMPLE_PAGE_2), pv(1, EXAMPLE_PAGE_1))
        self.assertEqual(led.chains, ((1, 2),))
        self.assertEqual(led.rows(), HAND_ROWS)

    def test_prev_page_inheritance_resolves_to_true_origin(self):
        led = ledger_of(pv(1, EXAMPLE_PAGE_1), pv(2, EXAMPLE_PAGE_2))
        n3 = led.by_seq(3)
        self.assertEqual(n3.group, "G")
        self.assertEqual(n3.worker, "A")
        self.assertEqual(n3.group_fill.source, "prev_page")
        self.assertEqual(n3.worker_fill.source, "prev_page")
        self.assertEqual(n3.group_fill.origin_page, 1)
        # The chain must pass through page 1's last row, not an adjacent page.
        self.assertEqual([link.page for link in n3.worker_fill.chain], [1, 2])

    def test_same_page_inheritance_after_first_line(self):
        led = ledger_of(pv(1, EXAMPLE_PAGE_1))
        n2 = led.by_seq(2)
        self.assertEqual(n2.group_fill.source, "same_page")
        self.assertEqual(n2.group, "G")

    def test_byte_ranges_reconstruct_the_actual_token(self):
        body = encode_text_bytes(EXAMPLE_PAGE_1)
        led = ledger_of(PageVersion(1, 0, body))
        n1 = led.by_seq(1)
        g = n1.group_fill
        self.assertEqual(body[g.origin_span.start : g.origin_span.end], b"G")
        w = n1.worker_fill
        self.assertEqual(body[w.origin_span.start : w.origin_span.end], b"A")
        # The full page text is reproducible from the decoded bytes.
        self.assertEqual(decode_document(body).text, EXAMPLE_PAGE_1)

    def test_equals_on_start_page_first_line_rejected(self):
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(pv(1, "PAGE|1|-\n1|=|A|YES\n"))
        self.assertEqual(ctx.exception.kind, "inherit_without_source")

    def test_response_codes_are_exact_ascii(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|YES\n2|G|B|NO\n3|G|C|MISS\n"))
        self.assertEqual([n.response for n in led.notices], ["YES", "NO", "MISS"])

    def test_bad_response_rejected(self):
        for bad in ("yes", "NO ", "MAYBE", "=", ""):
            with self.assertRaises(LedgerError) as ctx:
                ledger_of(pv(1, f"PAGE|1|-\n1|G|A|{bad}\n"))
            self.assertIn(ctx.exception.kind, ("bad_response", "bad_record"))

    def test_duplicate_sequence_rejected(self):
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(
                pv(1, "PAGE|1|-\n1|G|A|YES\n"),
                pv(2, "PAGE|2|1\n1|G|B|NO\n"),
            )
        self.assertEqual(ctx.exception.kind, "duplicate_sequence")


class TrapTests(unittest.TestCase):
    def test_pipe_low_byte_inside_pair_is_not_a_separator(self):
        sample = build_sample()
        # The worker U+4F7C is encoded as 01 7C: byte 7C is data.
        page2 = next(p for p in sample.pages if p.page == 2)
        self.assertIn(bytes((0x01, 0x7C)), page2.body)
        # A byte splitter would see 5 fields; the character splitter sees 4.
        led = ledger_from_versions(sample.pages)
        n4 = led.by_seq(4)
        self.assertEqual(n4.worker, WORKER_WITH_7C)
        self.assertEqual(n4.group, "H")
        self.assertEqual(n4.worker_fill.origin_span.end - n4.worker_fill.origin_span.start, 2)

    def test_noop_switch_is_not_a_separator(self):
        sample = build_sample()
        page1 = next(p for p in sample.pages if p.page == 1)
        self.assertIn(bytes((0xFF, 0x00)), page1.body)
        # Decoded text has no extra separator, so parsing still sees 4 fields.
        self.assertEqual(decode_document(page1.body).text, EXAMPLE_PAGE_1)
        led = ledger_from_versions(sample.pages)
        self.assertEqual(led.rows()[0], (1, "G", "A", "YES"))

    def test_same_bytes_split_differently_by_naive_byte_reader(self):
        # Demonstrate the trap concretely: a naive bytes.split(b"|") on the
        # page-2 line with U+4F7C yields an extra field, while the codec
        # parser does not.
        sample = build_sample()
        page2 = next(p for p in sample.pages if p.page == 2)
        raw_line = [ln for ln in page2.body.split(b"\n") if ln.startswith(b"4|")][0]
        self.assertEqual(len(raw_line.split(b"|")), 5)  # phantom column
        text_line = [ln for ln in decode_document(page2.body).text.split("\n") if ln.startswith("4|")][0]
        self.assertEqual(len(text_line.split("|")), 4)


class ChainTests(unittest.TestCase):
    def test_missing_predecessor_reports_affected_pages(self):
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(pv(3, "PAGE|3|2\n5|=|=|NO\n"))
        self.assertEqual(ctx.exception.kind, "missing_predecessor")
        self.assertEqual(ctx.exception.pages, (2, 3))

    def test_missing_middle_page_breaks_the_chain(self):
        # page 2 is absent, so page 3 cannot inherit.
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(
                pv(1, EXAMPLE_PAGE_1),
                pv(3, "PAGE|3|2\n5|=|=|NO\n"),
            )
        self.assertEqual(ctx.exception.kind, "missing_predecessor")
        self.assertEqual(set(ctx.exception.pages), {2, 3})

    def test_cycle_reports_pages(self):
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(
                pv(1, "PAGE|1|2\n1|G|A|YES\n"),
                pv(2, "PAGE|2|1\n2|G|B|NO\n"),
            )
        self.assertEqual(ctx.exception.kind, "cycle")
        self.assertEqual(ctx.exception.pages, (1, 2))

    def test_fork_reports_both_successors(self):
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(
                pv(1, EXAMPLE_PAGE_1),
                pv(2, EXAMPLE_PAGE_2),
                pv(3, "PAGE|3|1\n9|G|X|YES\n"),
            )
        self.assertEqual(ctx.exception.kind, "fork")
        self.assertIn(1, ctx.exception.pages)
        self.assertTrue({2, 3} <= set(ctx.exception.pages))

    def test_duplicate_page_rejected(self):
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(pv(1, EXAMPLE_PAGE_1), pv(1, EXAMPLE_PAGE_1, version=1))
        self.assertEqual(ctx.exception.kind, "duplicate_page")

    def test_no_value_stolen_from_adjacent_start_page(self):
        # Page 10 legitimately continues page 1.  Page 9 is an unrelated
        # start page listed "next to" page 10 in the input; page 10 must take
        # G/A from page 1, never Z/Q from page 9.
        led = ledger_of(
            pv(1, EXAMPLE_PAGE_1),
            pv(9, "PAGE|9|-\n8|Z|Q|YES\n"),
            pv(10, "PAGE|10|1\n11|=|=|NO\n"),
        )
        n = led.by_seq(11)
        self.assertEqual((n.group, n.worker), ("G", "A"))
        self.assertEqual(n.group_fill.origin_page, 1)
        self.assertNotEqual(n.group, "Z")

    def test_declared_continuation_can_arrive_late(self):
        led = ledger_of(pv(2, EXAMPLE_PAGE_2), pv(1, EXAMPLE_PAGE_1))
        self.assertEqual(led.rows(), HAND_ROWS)


# ---------------------------------------------------------------------------
# Union roster


class RosterExampleTests(unittest.TestCase):
    """The exact example from the task statement."""

    def setUp(self):
        self.pages = pages(pv(1, "PAGE|1|-\n1|G|A|YES\n2|=|A|NO\n"))

    def test_second_call_is_disputed_and_expected_b(self):
        led = ledger_of(*self.pages)
        roster = roster_of({"G": ("A", "B", "C")})
        calls = roster.replay(led.notices)
        self.assertEqual(calls[0].expected, "A")
        self.assertFalse(calls[0].disputed)
        self.assertEqual(calls[1].expected, "B")
        self.assertTrue(calls[1].disputed)
        self.assertEqual(calls[1].actual, "A")
        # Queue B, C, A; the wrong call left the turn untouched.
        self.assertEqual(calls[1].order_after, ("B", "C", "A"))

    def test_correcting_first_response_to_miss_makes_second_call_valid(self):
        corrected = pv(1, "PAGE|1|-\n1|G|A|MISS\n2|=|A|NO\n", version=1)
        led = ledger_of(corrected)
        roster = roster_of({"G": ("A", "B", "C")})
        calls = roster.replay(led.notices)
        # MISS leaves A at the front, so A is expected and A was called.
        self.assertEqual(calls[0].order_after, ("A", "B", "C"))
        self.assertEqual(calls[1].expected, "A")
        self.assertFalse(calls[1].disputed)
        self.assertEqual(calls[1].order_after, ("B", "C", "A"))

    def test_fixed_yes_with_b_and_c_on_leave_second_call_is_a(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|YES\n2|=|A|NO\n"))
        roster = roster_of(
            {"G": ("A", "B", "C")},
            leaves={"G": [Leave("B", 2, 3), Leave("C", 2, 3)]},
        )
        calls = roster.replay(led.notices)
        # A already rotated to the tail after YES; B and C are on leave at 2.
        self.assertEqual(calls[1].expected, "A")
        self.assertFalse(calls[1].disputed)

    def test_leave_is_half_open(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|YES\n2|=|A|NO\n"))
        # B on leave [2,3): ineligible at 2, eligible at 3.
        roster = roster_of({"G": ("A", "B", "C")}, leaves={"G": [Leave("B", 2, 3)]})
        calls = roster.replay(led.notices)
        # After A's YES the queue is B,C,A; at time 2 B is on leave so C is
        # the expected caller (A actually called, so this is a dispute).
        self.assertEqual(calls[1].expected, "C")
        self.assertEqual(dict(calls[1].skipped)["B"], "on leave [2,3)")
        # At time 2, B is out but C is the expected first eligible.
        self.assertEqual(roster.who_at(led.notices, 2)["G"], "C")
        # At time 3 (half-open end), B is back and still at the head.
        self.assertEqual(roster.who_at(led.notices, 3)["G"], "B")

    def test_yes_and_no_rotate_miss_stays(self):
        led = ledger_of(
            pv(1, "PAGE|1|-\n1|G|A|YES\n2|G|B|NO\n3|G|A|MISS\n"),
        )
        roster = roster_of({"G": ("A", "B", "C")})
        calls = roster.replay(led.notices)
        # 1: A YES -> B,C,A.  2: B NO -> C,A,B.  3: expected C, actual A is
        # a dispute (no rotation).  To test MISS cleanly, redo with correct A.
        self.assertEqual(calls[0].order_after, ("B", "C", "A"))
        self.assertEqual(calls[1].order_after, ("C", "A", "B"))
        self.assertEqual(calls[2].expected, "C")
        self.assertTrue(calls[2].disputed)

    def test_miss_on_correct_worker_keeps_position(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|MISS\n2|G|A|YES\n"))
        roster = roster_of({"G": ("A", "B", "C")})
        calls = roster.replay(led.notices)
        self.assertEqual(calls[0].order_after, ("A", "B", "C"))
        self.assertFalse(calls[0].disputed)
        self.assertEqual(calls[1].order_after, ("B", "C", "A"))

    def test_ineligible_stays_and_returns_to_original_rank(self):
        led = ledger_of(
            pv(1, "PAGE|1|-\n1|G|A|YES\n2|G|C|YES\n3|G|A|YES\n4|G|B|YES\n"),
        )
        # B is ineligible only for seq 2 and 3 (half-open ends at 4).
        roster = roster_of(
            {"G": ("A", "B", "C")},
            leaves={"G": [Leave("B", 2, 4)]},
        )
        calls = roster.replay(led.notices)
        # seq1 A rotates -> B,C,A.  seq2: B on leave, C called -> B,A,C.
        self.assertEqual(calls[1].expected, "C")
        self.assertFalse(calls[1].disputed)
        self.assertEqual(calls[1].order_after, ("B", "A", "C"))
        # seq3: B still on leave; A called -> B,C,A.
        self.assertEqual(calls[2].expected, "A")
        self.assertFalse(calls[2].disputed)
        # seq4: B is back and still ahead in the queue -> B expected.
        self.assertEqual(calls[3].expected, "B")
        self.assertFalse(calls[3].disputed)

    def test_qualified_from_restores_position(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|YES\n2|G|C|YES\n"))
        roster = roster_of(
            {"G": ("A", "B", "C")},
            qualified_from={"G": {"B": 2}},
        )
        calls = roster.replay(led.notices)
        # At seq 2, B is now qualified (>=2) and ahead in turn, so B expected.
        self.assertEqual(calls[1].expected, "B")
        self.assertTrue(calls[1].disputed)

    def test_no_eligible_worker_is_an_error(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|YES\n"))
        roster = roster_of(
            {"G": ("A", "B", "C")},
            leaves={"G": [Leave("A", 1, 2), Leave("B", 1, 2), Leave("C", 1, 2)]},
        )
        with self.assertRaises(RosterError):
            roster.replay(led.notices)

    def test_duplicate_rotation_name_rejected(self):
        with self.assertRaises(RosterError):
            GroupRoster("G", ("A", "A", "B"))


class IndependentQueryTests(unittest.TestCase):
    def setUp(self):
        self.led = ledger_of(
            pv(1, "PAGE|1|-\n1|G|A|YES\n2|G|B|NO\n3|G|A|MISS\n"),
        )
        self.roster = roster_of({"G": ("A", "B", "C")})

    def test_who_at_is_independent_of_query_order(self):
        # Query a later time first, then an earlier one: results must not
        # depend on any hidden shared state.
        later = self.roster.who_at(self.led.notices, 3)
        earlier = self.roster.who_at(self.led.notices, 2)
        self.assertEqual(earlier["G"], "B")
        self.assertEqual(later["G"], "C")
        # Repeat to prove determinism.
        self.assertEqual(self.roster.who_at(self.led.notices, 3), later)
        self.assertEqual(self.roster.who_at(self.led.notices, 2), earlier)

    def test_disputes_at_prefix(self):
        all_disputes = self.roster.disputes_at(self.led.notices)
        first_two = self.roster.disputes_at(self.led.notices, up_to=2)
        self.assertEqual([d.seq for d in all_disputes], [3])
        self.assertEqual(first_two, [])
        # Each dispute records the expected caller.
        self.assertEqual(all_disputes[0].expected, "C")
        self.assertEqual(all_disputes[0].actual, "A")

    def test_order_at_and_after(self):
        self.assertEqual(self.roster.order_at(self.led.notices, 1)["G"], ("A", "B", "C"))
        self.assertEqual(self.roster.order_after(self.led.notices)["G"], ("C", "A", "B"))

    def test_replay_is_repeatable(self):
        first = [c.order_after for c in self.roster.replay(self.led.notices)]
        second = [c.order_after for c in self.roster.replay(self.led.notices)]
        self.assertEqual(first, second)


class RosterDoesNotTouchUnionOrderTests(unittest.TestCase):
    def test_actual_calls_never_reorder_rotation_except_on_match(self):
        # Three consecutive wrong calls must leave the union order intact.
        led = ledger_of(
            pv(1, "PAGE|1|-\n1|G|C|YES\n2|G|C|YES\n3|G|C|YES\n"),
        )
        roster = roster_of({"G": ("A", "B", "C")})
        calls = roster.replay(led.notices)
        for call in calls:
            self.assertTrue(call.disputed)
            self.assertEqual(call.expected, "A")
            self.assertEqual(call.order_after, ("A", "B", "C"))

    def test_rotation_is_original_configuration(self):
        roster = roster_of({"G": ("A", "B", "C")})
        self.assertEqual(roster.group("G").rotation, ("A", "B", "C"))


# ---------------------------------------------------------------------------
# Frozen originals: correction replay and pinned re-query


class DisputeReplayTests(unittest.TestCase):
    def setUp(self):
        self.sample = build_sample()
        self.roster = Roster(
            [
                GroupRoster(g, tuple(rot), frozenset(rot))
                for g, rot in self.sample.rotations.items()
            ]
        )

    def test_original_dispute_and_pinned_requery(self):
        old = ledger_from_versions(self.sample.pages)
        old_calls = {c.seq: c for c in self.roster.replay(old.notices)}
        # G's second call (seq 2) and continuation (seq 3) are disputed.
        self.assertTrue(old_calls[2].disputed)
        self.assertEqual(old_calls[2].expected, "B")
        self.assertEqual(old_calls[3].expected, "B")

        corrected = (
            PageVersion(1, 1, self.sample.corrected_page1),
            self.sample.pages[1],
            self.sample.pages[2],
        )
        new = ledger_from_versions(corrected)
        new_calls = {c.seq: c for c in self.roster.replay(new.notices)}
        # After the correction the second call is clean.
        self.assertFalse(new_calls[2].disputed)
        self.assertEqual(new_calls[2].expected, "A")
        # But page 3 continues page 2, whose last worker is still A, and page
        # 3's group is inherited from H, so seq 5 is unaffected by page 1.
        # The old page-1 version can still be re-queried byte-for-byte.
        self.assertEqual(
            ledger_from_versions((self.sample.pages[0],)).by_seq(1).response, "YES"
        )

    def test_dispute_replay_reports_changed_seqs(self):
        corrected = (
            PageVersion(1, 1, self.sample.corrected_page1),
            self.sample.pages[1],
            self.sample.pages[2],
        )
        result = dispute_replay(self.sample.pages, corrected, self.roster)
        changed = [c["seq"] for c in result["changed"]]
        self.assertIn(2, changed)
        # seq 3 depends on page 1's group via page 2, so a group change would
        # show it; a response-only correction leaves page-2 rows unchanged.
        for item in result["changed"]:
            self.assertIn("old", item)
            self.assertIn("new", item)

    def test_correction_affects_dependent_notice_not_just_its_page(self):
        # Change the *group* at seq 2 on page 1.  Page 2's first line inherits
        # the group from page 1, so its expected caller must follow.
        corrected_page1 = encode_text_bytes(
            "PAGE|1|-\n1|G|A|YES\n2|H|A|NO\n"
        )
        corrected = (
            PageVersion(1, 1, corrected_page1),
            self.sample.pages[1],
            self.sample.pages[2],
        )
        old = ledger_from_versions(self.sample.pages)
        new = ledger_from_versions(corrected)
        self.assertEqual(old.by_seq(3).group, "G")
        self.assertEqual(new.by_seq(3).group, "H")
        self.assertEqual(new.by_seq(3).group_fill.source, "prev_page")
        result = dispute_replay(self.sample.pages, corrected, self.roster)
        self.assertIn(3, [c["seq"] for c in result["changed"]])

    def test_ledger_book_pins_versions(self):
        from solo.ledger import LedgerBook

        book = LedgerBook()
        for page in self.sample.pages:
            book.add_page(page.page, page.body)
        # Apply a real correction: replace "YES" with "MISS" on page 1.
        from solo.revision import replace

        text = decode_document(book.version(1, 0).body).text
        start = text.index("YES")
        corrected = book.apply(1, 0, [replace(start, start + 3, "MISS")])
        self.assertEqual(corrected.version_id, 1)

        latest = book.ledger()
        self.assertEqual(latest.by_seq(1).response, "MISS")
        pinned = book.ledger(select={1: 0})
        self.assertEqual(pinned.by_seq(1).response, "YES")
        # Old bytes remain intact.
        self.assertEqual(decode_document(book.version(1, 0).body).text.split("\n")[1], "1|G|A|YES")

    def test_format_ledger_shows_source_chain_and_range(self):
        led = ledger_from_versions(self.sample.pages)
        lines = format_ledger(led)
        joined = "\n".join(lines)
        self.assertIn("prev_page", joined)
        self.assertIn("chain", joined)
        self.assertIn("origin p1v0", joined)


# ---------------------------------------------------------------------------
# Missing page and reinsertion replay


class MissingAndReinsertionTests(unittest.TestCase):
    def setUp(self):
        self.sample = build_sample()
        self.by_page = {p.page: p for p in self.sample.pages}

    def test_page_dropped_then_reinserted_gives_identical_rows(self):
        full = ledger_from_versions(self.sample.pages)
        # Drop page 2: pages 1 and 3 now form two chains, and page 3's
        # declared predecessor is missing -> the chain is rejected.
        with self.assertRaises(LedgerError) as ctx:
            ledger_from_versions((self.by_page[1], self.by_page[3]))
        self.assertEqual(ctx.exception.kind, "missing_predecessor")
        self.assertEqual(ctx.exception.pages, (2, 3))

        # Reinsert page 2: identical notices and chains to the full book.
        restored = ledger_from_versions(
            (self.by_page[1], self.by_page[2], self.by_page[3])
        )
        self.assertEqual(restored.rows(), full.rows())
        self.assertEqual(restored.chains, full.chains)
        self.assertEqual(
            [n.worker_fill.source for n in restored.notices],
            [n.worker_fill.source for n in full.notices],
        )

    def test_dropping_a_middle_page_cannot_steal_across_chain(self):
        # Page 3 continues page 2.  Without page 2, page 3 must not inherit
        # page 1 just because it is the nearest available page.
        with self.assertRaises(LedgerError):
            ledger_from_versions((self.by_page[1], self.by_page[3]))
        # Even adding an unrelated start page does not let page 3 read it.
        with self.assertRaises(LedgerError):
            ledger_from_versions(
                (
                    self.by_page[1],
                    pv(7, "PAGE|7|-\n100|Z|Q|YES\n"),
                    self.by_page[3],
                )
            )

    def test_reinsertion_order_does_not_matter(self):
        a = ledger_from_versions((self.by_page[3], self.by_page[1], self.by_page[2]))
        b = ledger_from_versions((self.by_page[2], self.by_page[3], self.by_page[1]))
        self.assertEqual(a.rows(), b.rows())


# ---------------------------------------------------------------------------
# Encoding traps beyond the sample: PAIR separators that look like data


class EncodedTrapTests(unittest.TestCase):
    def test_pair_value_pipe_byte_in_group_and_worker(self):
        # Build a page whose group is a PAIR character with low byte 0x7C.
        body = encode_text_bytes(
            "PAGE|1|-\n1|" + WORKER_WITH_7C + "|" + WORKER_WITH_7C + "|YES\n"
        )
        self.assertIn(bytes((0x01, 0x7C)), body)
        led = ledger_from_versions((PageVersion(1, 0, body),))
        n = led.by_seq(1)
        self.assertEqual(n.group, WORKER_WITH_7C)
        self.assertEqual(n.worker, WORKER_WITH_7C)
        self.assertEqual(n.response, "YES")

    def test_parse_page_exposes_header_and_prev(self):
        parsed = parse_page(encode_text_bytes("PAGE|7|3\n9|G|A|YES\n"))
        self.assertEqual(parsed.page, 7)
        self.assertEqual(parsed.prev, 3)
        self.assertEqual(len(parsed.records), 1)

    def test_page_registry_mismatch_rejected(self):
        with self.assertRaises(LedgerError) as ctx:
            parse_page(encode_text_bytes("PAGE|2|1\n"), page=5)
        self.assertEqual(ctx.exception.kind, "page_mismatch")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class EmptyPageInheritanceTests(unittest.TestCase):
    def test_empty_page_cannot_supply_value_to_next(self):
        # Page 2 is a legitimate continuation (of page 1) but carries no
        # records.  Page 3 continues page 2 and writes '=' on its first line;
        # there is no last value on page 2, and values from page 1 must not be
        # borrowed.
        with self.assertRaises(LedgerError) as ctx:
            ledger_of(
                pv(1, EXAMPLE_PAGE_1),
                pv(2, "PAGE|2|1\n"),
                pv(3, "PAGE|3|2\n9|=|=|YES\n"),
            )
        self.assertEqual(ctx.exception.kind, "inherit_without_source")
        self.assertEqual(set(ctx.exception.pages), {2, 3})

    def test_empty_page_without_inheritance_is_fine(self):
        led = ledger_of(
            pv(1, EXAMPLE_PAGE_1),
            pv(2, "PAGE|2|1\n"),
            pv(3, "PAGE|3|2\n9|G|A|YES\n"),
        )
        self.assertEqual([n.seq for n in led.notices], [1, 2, 9])


class WhoAtRuleTests(unittest.TestCase):
    def test_who_at_skips_ineligible_head(self):
        led = ledger_of(pv(1, "PAGE|1|-\n1|G|A|YES\n"))
        roster = roster_of({"G": ("A", "B", "C")}, leaves={"G": [Leave("B", 5, 6)]})
        # After seq 1 (A YES) queue is B,C,A.  At time 5, B is on leave so the
        # very next call would be C, not the queue head.
        self.assertEqual(roster.who_at(led.notices, 5)["G"], "C")
        # At time 6, B is back.
        self.assertEqual(roster.who_at(led.notices, 6)["G"], "B")
