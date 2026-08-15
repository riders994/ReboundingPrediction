"""Locate each play-by-play miss in the tracking data.

Replaces the ``findRim`` / ``rimErr`` / ``findShot`` / ``rimshots`` group in
``coordinator.py``. For every missed shot the play-by-play reports, this finds two
tracking moments: the **release** (ball starts rising) and the **rim contact**
(ball arrives at the rim). Those bracket the window the models care about.

Correctness changes:

* **End-of-period shots no longer pair against the end of the period.** The old
  ``RimStart`` column was the clock on rim-contact rows and ``0`` everywhere else,
  so the filter ``RimStart >= t`` matched every row in the quarter whenever
  ``t == 0``, and ``.iloc[-1]`` then returned the quarter's last moment. Transitions
  are booleans now (see :mod:`rebounding.data.sportvu`), so the sentinel cannot be
  confused with a real clock value.

* **No fallback to an arbitrary moment.** When two shots resolved to the same
  tracking row the old code called ``rimErr``, which filtered on ``Clock >= t`` with
  no rim condition at all and returned whatever moment happened to be last. The
  result was stored without being checked for reuse, and an empty result was stored
  too, becoming a NaN row downstream. Unpairable shots are dropped and counted.

* **Rim contacts are assigned globally**, not claimed greedily in clock order, and
  made shots take part so misses cannot steal their rim contacts.

* **Release detection is bounded and anchored.** ``IsHighStart`` fires on every
  dribble bounce, every pass apex, and every rattle of the ball on the rim --
  9,073 times in the sample game against roughly 176 shots. The old ``findShot``
  took the most recent one before rim contact with no bound at all. ``DT`` was
  already computed but only ever used as a model feature, never as a filter.

Pairing quality was previously unmeasurable. It is now checked by comparing the
tracking-derived release distance against the distance stated in the play-by-play
description, which is why :mod:`rebounding.data.pbp` has to stop discarding the
visiting team's descriptions. Measured on the sample game (95 unblocked misses;
blocked shots are excluded because they never reach the rim):

===========================================  ========  ==========  =========
configuration                                  paired  median err  within 3ft
===========================================  ========  ==========  =========
original logic, ported as-is                       70     10.06 ft         29%
+ arc test on the release                          68      1.58 ft         70%
+ global assignment, looser rim arrival            79      1.61 ft         73%
+ near-rim rise relaxation                         87      1.58 ft         77%
+ shooter proximity required (current)             78      1.31 ft         84%
===========================================  ========  ==========  =========

The last step trades ten percent of recall for a sizeable accuracy gain, which is
the right way round: a mispaired shot is a corrupted training row, and with 636
games available the shots are cheaper than the corruption.

Every dropped shot is recorded in :class:`PairingReport` with a reason, so a
636-game rebuild can be audited instead of silently losing games.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from rebounding.data.sportvu import GameTracking

# A shot's flight from release to the rim. Outside this the pairing is spurious.
MIN_FLIGHT_SECONDS = 0.3
MAX_FLIGHT_SECONDS = 3.0

# Rim contact happens at or shortly before the play-by-play records the miss. The
# clock runs down, so an earlier event has a *larger* clock value.
RIM_SEARCH_BEFORE_SECONDS = 4.0
RIM_SEARCH_AFTER_SECONDS = 1.0

# A shot's arc, used to tell a release from a dribble or from rim jitter. Anything
# that reaches the rim must pass above it, and must have climbed to get there.
MIN_SHOT_PEAK_FT = 10.5
MIN_SHOT_RISE_FT = 4.0

# How close the ball must be to the named shooter for a frame to count as the
# release. Wide enough for tracking noise and for the ball already being out of the
# hand by the time the frame is sampled.
MAX_RELEASE_SHOOTER_DISTANCE_FT = 5.0

# A dunk leaves the hand essentially at the rim, so the release-to-rim gap can be
# far shorter than a jump shot's flight. Only reject a zero-length gap.
# Inside this range of the basket a shot is a layup, dunk or tip, which releases
# near rim height and barely climbs.
NEAR_RIM_SHOT_FT = 10.0
MIN_NEAR_RIM_RISE_FT = 1.0

DROP_NO_RIM_CONTACT = "no_rim_contact_near_reported_clock"
DROP_RIM_ALREADY_USED = "rim_contact_already_claimed"
DROP_NO_RELEASE = "no_release_within_flight_bounds"


@dataclass
class PairingReport:
    """Why shots were kept or dropped, for auditing a bulk rebuild."""

    game_id: str
    n_shots: int = 0
    n_paired: int = 0
    drops: Counter = field(default_factory=Counter)

    @property
    def pair_rate(self) -> float:
        return self.n_paired / self.n_shots if self.n_shots else 0.0

    def __str__(self) -> str:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(self.drops.items())) or "none"
        return (
            f"{self.game_id}: paired {self.n_paired}/{self.n_shots} "
            f"({self.pair_rate:.1%}); drops: {detail}"
        )


def _quarter_candidates(moments: pd.DataFrame, column: str) -> dict[int, np.ndarray]:
    """Positional indices where ``column`` is true, grouped by quarter."""
    flagged = moments.index[moments[column]].to_numpy()
    quarters = moments["Quarter"].to_numpy()
    return {int(q): flagged[quarters[flagged] == q] for q in np.unique(quarters[flagged])}


def assign_rim_contacts(
    shot_clocks: np.ndarray,
    rim_starts: np.ndarray,
    rim_clocks: np.ndarray,
) -> dict[int, int]:
    """Match shots to rim contacts within one quarter, minimising total clock error.

    Returns ``{shot position: rim contact index}`` for the shots that matched.

    A global assignment rather than the original greedy scan. Greedily letting each
    shot take its nearest unclaimed rim contact loses shots whenever an earlier one
    takes a contact that a later one needed -- 36 of 106 in the sample game. The
    Hungarian assignment is also why made shots have to be passed in: they generate
    rim contacts too, and if misses can claim those, both end up wrong.
    """
    if shot_clocks.size == 0 or rim_starts.size == 0:
        return {}

    delta = np.abs(rim_clocks[None, :] - shot_clocks[:, None])
    allowed = (rim_clocks[None, :] >= shot_clocks[:, None] - RIM_SEARCH_AFTER_SECONDS) & (
        rim_clocks[None, :] <= shot_clocks[:, None] + RIM_SEARCH_BEFORE_SECONDS
    )
    # linear_sum_assignment cannot take infinities, so disallowed pairs get a cost
    # far above any allowed one and are filtered out of the result.
    forbidden = delta.max() + 1.0 if delta.size else 1.0
    cost = np.where(allowed, delta, forbidden)

    rows, cols = linear_sum_assignment(cost)
    return {
        int(r): int(rim_starts[c])
        for r, c in zip(rows, cols, strict=True)
        if allowed[r, c]
    }


def find_release(
    moments: pd.DataFrame,
    high_starts: np.ndarray,
    rim_idx: int,
    ball_xyz: np.ndarray,
    rim_distances: np.ndarray,
    shooter_xy: np.ndarray | None = None,
) -> int | None:
    """Index of the release preceding ``rim_idx``, or ``None`` if none is plausible.

    A plausible flight time is necessary but nowhere near sufficient. ``IsHighStart``
    fires on any frame where the ball stops descending, which includes dribbles,
    pass apexes, and -- the case that actually broke this -- tracking jitter while
    the ball sits at the rim. On a measured 24-foot three the ball left the
    shooter's hands 2.39 s before rim contact, but three spurious high starts fired
    within 0.5 s of the rim as the ball rattled, each of them inside the flight-time
    bounds.

    Two filters, applied in order.

    The arc: anything that reaches the rim had to climb and had to pass above it.
    Rim jitter rises under a foot; a dribble bounce rises three or four feet and
    never gets near rim height.

    The shooter: the play-by-play names who took the shot and the tracking data
    carries player ids, so the release is where the ball is *next to that player*.
    Pass ``shooter_xy`` (the shooter's position at every moment) to use it. This is
    what separates an alley-oop lob from the dunk that follows -- both are clean
    arcs ending at the rim, but only one starts in the shooter's hands.
    """
    if high_starts.size == 0:
        return None

    clocks = moments["GameClock"].to_numpy()
    ball_z = ball_xyz[:, 2]
    rim_clock = clocks[rim_idx]

    before = high_starts[high_starts < rim_idx]
    if before.size == 0:
        return None

    flight = clocks[before] - rim_clock
    candidates = before[(flight >= MIN_FLIGHT_SECONDS) & (flight <= MAX_FLIGHT_SECONDS)]

    # Arc test. A layup or dunk releases at shoulder height a few feet from the
    # basket and barely climbs, so the required rise relaxes near the rim -- with
    # the flat threshold those shots fell through to the pass that preceded them.
    arc_ok = []
    for idx in candidates:
        peak = ball_z[idx : rim_idx + 1].max()
        if peak < MIN_SHOT_PEAK_FT:
            continue
        near_basket = rim_distances[idx] <= NEAR_RIM_SHOT_FT
        required_rise = MIN_NEAR_RIM_RISE_FT if near_basket else MIN_SHOT_RISE_FT
        if peak - ball_z[idx] >= required_rise:
            arc_ok.append(int(idx))

    if not arc_ok:
        return None

    # Prefer a release that is actually in the named shooter's hands. This is what
    # separates an alley-oop lob from the dunk that follows: both are clean arcs
    # ending at the rim, but only one starts with the shooter.
    if shooter_xy is not None:
        idx_arr = np.asarray(arc_ok)
        to_shooter = np.hypot(
            ball_xyz[idx_arr, 0] - shooter_xy[idx_arr, 0],
            ball_xyz[idx_arr, 1] - shooter_xy[idx_arr, 1],
        )
        near_shooter = idx_arr[to_shooter <= MAX_RELEASE_SHOOTER_DISTANCE_FT]
        # When the shooter is tracked this is a requirement, not a preference. The
        # near-rim rise relaxation above lets rim jitter back into `arc_ok`, and
        # falling through to the latest candidate would hand a 24-foot three a
        # release three feet from the basket. Dropping the shot beats recording a
        # position we know is wrong.
        return int(near_shooter[-1]) if near_shooter.size else None

    # Shooter untracked: the latest arc-plausible candidate, so an earlier pass in
    # the same window does not beat the shot itself.
    return arc_ok[-1]


def shooter_positions(tracking: GameTracking, shooter_id: int | None) -> np.ndarray | None:
    """The named player's ``(x, y)`` at every moment, or ``None`` if untracked.

    Returns NaN at moments where the player is off the floor, which propagates to a
    NaN distance and simply fails the proximity test.
    """
    if shooter_id is None or pd.isna(shooter_id):
        return None
    match = tracking.player_ids == int(shooter_id)
    if not match.any():
        return None

    out = np.full((len(tracking), 2), np.nan, dtype=np.float32)
    rows, cols = np.nonzero(match)
    out[rows] = tracking.player_xyz[rows, cols, :2]
    return out


def pair(
    tracking: GameTracking,
    shots: pd.DataFrame,
    made: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, PairingReport]:
    """Attach release and rim-contact tracking indices to each play-by-play miss.

    ``shots`` is the frame from :func:`rebounding.data.pbp.pair_shots_and_rebounds`.
    ``made`` comes from :func:`rebounding.data.pbp.made_shots`; those rows never
    become training data but they take part in the assignment so that misses cannot
    claim a made shot's rim contact.

    Returns the misses that could be located, with tracking columns added, plus a
    report of what was dropped and why.
    """
    moments = tracking.moments
    report = PairingReport(game_id=tracking.game_id, n_shots=len(shots))
    if shots.empty:
        return shots.copy(), report

    rim_by_quarter = _quarter_candidates(moments, "IsRimArrival")
    high_by_quarter = _quarter_candidates(moments, "IsHighStart")
    clocks = moments["GameClock"].to_numpy()
    ball_xyz = tracking.ball_xyz
    baskets = moments["Basket"].astype(str).to_numpy()
    rim_distances = moments["RimDistance"].to_numpy()
    shooter_cache: dict[int, np.ndarray | None] = {}

    misses = shots.reset_index(drop=True)
    made = made if made is not None else pd.DataFrame(columns=["Period", "Clock"])

    records = []
    for quarter, quarter_misses in misses.groupby("Period", sort=True):
        rim_starts = rim_by_quarter.get(int(quarter), np.empty(0, dtype=int))
        high_starts = high_by_quarter.get(int(quarter), np.empty(0, dtype=int))

        if rim_starts.size == 0:
            report.drops[DROP_NO_RIM_CONTACT] += len(quarter_misses)
            continue

        # Misses first, then makes, so a returned row index below the miss count
        # identifies which miss it belongs to.
        quarter_made = made[made["Period"] == quarter]
        shot_clocks = np.concatenate(
            [
                quarter_misses["Clock"].to_numpy(dtype=float),
                quarter_made["Clock"].to_numpy(dtype=float),
            ]
        )
        assignment = assign_rim_contacts(shot_clocks, rim_starts, clocks[rim_starts])

        for local_pos, (_, shot) in enumerate(quarter_misses.iterrows()):
            rim_idx = assignment.get(local_pos)
            if rim_idx is None:
                report.drops[DROP_NO_RIM_CONTACT] += 1
                continue

            shooter_id = shot.get("ShootPlayerID")
            key = int(shooter_id) if pd.notna(shooter_id) else -1
            if key not in shooter_cache:
                shooter_cache[key] = shooter_positions(tracking, shooter_id)

            release_idx = find_release(
                moments, high_starts, rim_idx, ball_xyz, rim_distances, shooter_cache[key]
            )
            if release_idx is None:
                report.drops[DROP_NO_RELEASE] += 1
                continue

            records.append(
                {
                    **shot.to_dict(),
                    "ReleaseIdx": release_idx,
                    "RimIdx": rim_idx,
                    "ReleaseClock": float(clocks[release_idx]),
                    "RimClock": float(clocks[rim_idx]),
                    "FlightTime": float(clocks[release_idx] - clocks[rim_idx]),
                    # Resolved once per shot from the ball at the rim, then applied
                    # to everyone. See rebounding.data.court on why this must not be
                    # a per-player decision.
                    "Basket": baskets[rim_idx],
                }
            )

    report.n_paired = len(records)
    paired = pd.DataFrame.from_records(records)
    if not paired.empty:
        paired = paired.sort_values(["Period", "Clock"], ascending=[True, False]).reset_index(drop=True)
    return paired, report
