"""Union duty roster, leave and "who is next" resolution.

The union keeps its *own* starting rotation per group.  A group's rotation is
a list of distinct worker names; the first name is the starting turn.  The
ledger (see :mod:`solo.ledger`) only records what the shop *actually* did; it
never changes the union order.  This module replays notifications against the
union roster and reports the candidate who *should* have been called.

Rules
-----
For a notification at time ``t`` for group ``g``:

* eligibility is "qualified at ``t``" and "not on leave at ``t``"; leave
  intervals are half-open ``[start, end)``;
* scan the group queue from the head and take the first eligible worker;
* if that worker is not the one actually called, record a **dispute** with the
  expected worker, and leave the rotation untouched (the turn carries over to
  the next notification);
* on a match: ``YES`` and ``NO`` both move the worker to the tail; ``MISS``
  leaves the worker in place;
* a worker who is temporarily ineligible stays at their position; when they
  become eligible again they are still in turn.

Every public query (:meth:`Roster.replay`, :meth:`Roster.who_at`,
:meth:`Roster.disputes_at`, :meth:`Roster.order_at`) is independent: it
recomputes the queue from the start of time, so results never depend on which
query was run first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .ledger import Notice

__all__ = [
    "Leave",
    "RosterError",
    "Dispute",
    "Call",
    "Roster",
    "GroupRoster",
]


class RosterError(Exception):
    """Invalid roster configuration or query."""


# ---------------------------------------------------------------------------
# Configuration


@dataclass(frozen=True)
class Leave:
    """Half-open leave interval ``[start, end)`` for one worker."""

    worker: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start >= self.end:
            raise RosterError(
                f"leave for {self.worker!r} must satisfy start < end "
                f"(got [{self.start},{self.end}))"
            )

    def covers(self, time: int) -> bool:
        return self.start <= time < self.end


@dataclass(frozen=True)
class GroupRoster:
    """One group's distinct starting rotation and its workers' properties."""

    group: str
    rotation: tuple[str, ...]
    qualified: frozenset[str] = frozenset()
    qualified_from: Mapping[str, int] = field(default_factory=dict)
    leaves: tuple[Leave, ...] = ()

    def __post_init__(self) -> None:
        if not self.rotation:
            raise RosterError(f"group {self.group!r} has an empty rotation")
        if len(set(self.rotation)) != len(self.rotation):
            seen: set[str] = set()
            dupes: list[str] = []
            for name in self.rotation:
                if name in seen:
                    dupes.append(name)
                seen.add(name)
            raise RosterError(
                f"group {self.group!r} starting rotation has duplicate names {dupes}"
            )
        roster_names = set(self.rotation)
        unknown = sorted(set(self.qualified) - roster_names)
        if unknown:
            raise RosterError(
                f"group {self.group!r} qualifies workers not in the rotation: {unknown}"
            )
        unknown = sorted(set(self.qualified_from) - roster_names)
        if unknown:
            raise RosterError(
                f"group {self.group!r} gives qualified_from for non-rotation "
                f"workers: {unknown}"
            )
        for leave in self.leaves:
            if leave.worker not in roster_names:
                raise RosterError(
                    f"group {self.group!r} has leave for non-rotation worker "
                    f"{leave.worker!r}"
                )

    def is_eligible(self, worker: str, time: int) -> bool:
        if not self._qualified(worker, time):
            return False
        return not self._on_leave(worker, time)

    def _qualified(self, worker: str, time: int) -> bool:
        if worker in self.qualified:
            return True
        since = self.qualified_from.get(worker)
        if since is None:
            return False
        return time >= since

    def _on_leave(self, worker: str, time: int) -> bool:
        return any(leave.worker == worker and leave.covers(time) for leave in self.leaves)

    def ineligible_reason(self, worker: str, time: int) -> str | None:
        """Why ``worker`` cannot be called at ``time`` (``None`` if eligible)."""

        if not self._qualified(worker, time):
            since = self.qualified_from.get(worker)
            if since is not None:
                return f"not yet qualified (from {since})"
            return "not qualified"
        for leave in self.leaves:
            if leave.worker == worker and leave.covers(time):
                return f"on leave [{leave.start},{leave.end})"
        return None


def _as_roster(item: GroupRoster | tuple) -> GroupRoster:
    if isinstance(item, GroupRoster):
        return item
    group, rotation, *rest = item
    kwargs: dict = {}
    if rest:
        qualified, *rest2 = rest
        kwargs["qualified"] = frozenset(qualified)
        if rest2:
            kwargs["leaves"] = tuple(rest2[0])
    return GroupRoster(group=group, rotation=tuple(rotation), **kwargs)


# ---------------------------------------------------------------------------
# Results


@dataclass(frozen=True)
class Call:
    """Outcome of replaying one notification against the union roster."""

    notice: Notice
    group: str
    actual: str
    expected: str
    response: str
    eligible: bool
    disputed: bool
    scanned: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    order_before: tuple[str, ...]
    order_after: tuple[str, ...]

    @property
    def seq(self) -> int:
        return self.notice.seq

    @property
    def correct(self) -> bool:
        return not self.disputed and self.eligible


@dataclass(frozen=True)
class Dispute:
    seq: int
    group: str
    actual: str
    expected: str
    response: str
    scanned: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]

    def as_tuple(self) -> tuple[int, str, str, str]:
        return (self.seq, self.group, self.actual, self.expected)


# ---------------------------------------------------------------------------
# Roster


class Roster:
    """Union rosters plus a deterministic replay of a ledger."""

    def __init__(self, groups: Iterable[GroupRoster | tuple]) -> None:
        self._groups: dict[str, GroupRoster] = {}
        for item in groups:
            roster = _as_roster(item)
            if roster.group in self._groups:
                raise RosterError(f"group {roster.group!r} configured twice")
            self._groups[roster.group] = roster
        if not self._groups:
            raise RosterError("roster has no groups")

    def group(self, name: str) -> GroupRoster:
        try:
            return self._groups[name]
        except KeyError:
            raise RosterError(f"unknown group {name!r}") from None

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._groups))

    # -- replay -----------------------------------------------------------

    def replay(self, notices: Sequence[Notice]) -> list[Call]:
        """Replay notices in sequence order, one shared queue per group.

        Each call sees the group queue left by the previous call *in sequence
        order*.  The result is a fresh list; repeated calls recompute from the
        original rotations.
        """

        ordered = sorted(notices, key=lambda n: n.seq)
        queues: dict[str, list[str]] = {
            name: list(roster.rotation) for name, roster in self._groups.items()
        }
        calls: list[Call] = []
        for notice in ordered:
            try:
                roster = self._groups[notice.group]
            except KeyError:
                raise RosterError(
                    f"notice seq {notice.seq} names unknown group {notice.group!r}"
                ) from None
            calls.append(self._replay_one(roster, notice, queues[roster.group]))
        return calls

    def _replay_one(
        self, roster: GroupRoster, notice: Notice, queue: list[str]
    ) -> Call:
        time = notice.seq
        order_before = tuple(queue)
        scanned: list[str] = []
        skipped: list[tuple[str, str]] = []
        expected: str | None = None
        for worker in queue:
            scanned.append(worker)
            reason = roster.ineligible_reason(worker, time)
            if reason is None:
                expected = worker
                break
            skipped.append((worker, reason))
        if expected is None:
            raise RosterError(
                f"seq {notice.seq}: no eligible worker in group {roster.group!r} "
                f"(scanned {list(scanned)})"
            )

        disputed = notice.worker != expected
        if not disputed:
            # Only a correctly-called worker moves: YES/NO to the tail, MISS
            # stays in place.
            if notice.response in ("YES", "NO"):
                queue.remove(expected)
                queue.append(expected)
        order_after = tuple(queue)
        return Call(
            notice=notice,
            group=roster.group,
            actual=notice.worker,
            expected=expected,
            response=notice.response,
            eligible=not roster.ineligible_reason(expected, time),
            disputed=disputed,
            scanned=tuple(scanned),
            skipped=tuple(skipped),
            order_before=order_before,
            order_after=order_after,
        )

    # -- independent queries ----------------------------------------------

    def who_at(self, notices: Sequence[Notice], time: int, group: str | None = None) -> dict[str, str]:
        """First eligible caller per group at ``time``.

        Only notices with ``seq < time`` are replayed to build the queue, then
        the group is scanned at ``time`` using that instant's qualification and
        leave.  This answers "who would be called next at ``time``"
        independently of any other query.
        """

        prior = [n for n in notices if n.seq < time]
        queues = self._queues_after(self.replay(prior))
        target = [group] if group is not None else list(self.groups)
        out: dict[str, str] = {}
        for g in target:
            roster = self.group(g)
            chosen = None
            for worker in queues[g]:
                if roster.is_eligible(worker, time):
                    chosen = worker
                    break
            if chosen is None:
                raise RosterError(
                    f"no eligible worker in group {g!r} at time {time} "
                    f"(queue {tuple(queues[g])})"
                )
            out[g] = chosen
        return out

    def disputes_at(
        self, notices: Sequence[Notice], *, up_to: int | None = None
    ) -> list[Dispute]:
        """All disputes with ``seq <= up_to`` (default: all)."""

        selected = [n for n in notices if up_to is None or n.seq <= up_to]
        return [
            Dispute(
                seq=call.seq,
                group=call.group,
                actual=call.actual,
                expected=call.expected,
                response=call.response,
                scanned=call.scanned,
                skipped=call.skipped,
            )
            for call in self.replay(selected)
            if call.disputed
        ]

    def order_at(
        self, notices: Sequence[Notice], time: int, group: str | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Queue order per group just before ``time``."""

        prior = [n for n in notices if n.seq < time]
        queues = self._queues_after(self.replay(prior))
        target = [group] if group is not None else list(self.groups)
        return {g: tuple(queues[g]) for g in target}

    def order_after(
        self, notices: Sequence[Notice], *, group: str | None = None
    ) -> dict[str, tuple[str, ...]]:
        """Final queue order after replaying every notice."""

        queues = self._queues_after(self.replay(notices))
        target = [group] if group is not None else list(self.groups)
        return {g: tuple(queues[g]) for g in target}

    def _queues_after(self, calls: Sequence[Call]) -> dict[str, list[str]]:
        queues: dict[str, list[str]] = {
            name: list(roster.rotation) for name, roster in self._groups.items()
        }
        for call in calls:
            queues[call.group] = list(call.order_after)
        return queues
