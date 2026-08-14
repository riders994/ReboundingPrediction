"""Folding full-court SportVU coordinates onto a single half court.

The original pipeline did this in one line (``coordinator.py:204``)::

    frame['PlayerX'] = np.absolute(frame['PlayerX'].values - 47)

which has two distinct problems.

**It is a reflection, so court handedness flips.** For the right basket the
transform is ``x' = x - 47``; for the left it is ``x' = 47 - x`` with ``y``
untouched. The second one mirrors the court, so a play run down one sideline lands
on one side of the folded half court when it happens at one end and the opposite
side when it happens at the other. Left-wing and right-wing shots get superimposed
and a shot chart built from the folded data is meaningless.

The fix is to make the left-half transform a **180 degree rotation** rather than a
reflection, by flipping ``y`` along with ``x``. Rotation is orientation-preserving,
so the two ends stay consistent with each other.

**The attacking basket is never determined.** ``abs(x - 47)`` folds each player
independently according to that player's own ``x``, so a defender who has not
crossed half court is reflected into the offensive half. A player at ``x = 40``
during a shot at the right basket is genuinely 48.75 ft from the rim but folds to
``x' = 7`` and is recorded as 34.75 ft away. The basket has to be resolved once per
shot and the same transform applied to everyone.

Note on labelling: the transform guarantees the two ends are *consistent* with each
other, which is what shot charts and left/right features need. Which physical side
of the floor ends up at low ``y`` is a labelling question that depends on SportVU's
axis convention; resolve it empirically with :func:`describe_side_convention` and a
known player's shot chart rather than assuming.
"""

from __future__ import annotations

import numpy as np

from rebounding.constants import (
    CENTER_Y,
    COURT_WIDTH,
    HALF_COURT_X,
    HOOP,
    RIM_LEFT,
    RIM_RIGHT,
)

LEFT = "left"
RIGHT = "right"


def attacking_basket(ball_xy) -> str:
    """Return which basket the ball is at, as ``"left"`` or ``"right"``.

    Call this with the ball position at the rim-contact frame, where the answer is
    unambiguous: ``NearRim`` only fires when the ball is within a foot of a rim.
    Do not call it with a player position -- that is the bug this module exists to
    fix.
    """
    ball_xy = np.asarray(ball_xy, dtype=float)
    x, y = ball_xy[..., 0], ball_xy[..., 1]
    d_left = np.hypot(x - RIM_LEFT[0], y - RIM_LEFT[1])
    d_right = np.hypot(x - RIM_RIGHT[0], y - RIM_RIGHT[1])
    return LEFT if float(np.mean(d_left)) < float(np.mean(d_right)) else RIGHT


def fold(xy, basket: str):
    """Fold full-court ``(..., 2)`` coordinates onto the attacking half court.

    Both baskets map to :data:`rebounding.constants.HOOP` at ``(41.75, 25)``. The
    folded frame has ``x`` increasing from half court (0) toward the baseline (47),
    with ``y`` unchanged in range ``[0, 50]``.

    ``basket`` must come from :func:`attacking_basket` on the *ball*, and the same
    value must be used for all ten players and the ball on a given shot.
    """
    xy = np.asarray(xy, dtype=float)
    if xy.shape[-1] != 2:
        raise ValueError(f"expected trailing dimension of 2, got shape {xy.shape}")

    out = np.empty_like(xy)
    if basket == RIGHT:
        # Translate only. Orientation already matches the folded frame.
        out[..., 0] = xy[..., 0] - HALF_COURT_X
        out[..., 1] = xy[..., 1]
    elif basket == LEFT:
        # Rotate 180 degrees about center court, then translate. Flipping x alone
        # would be a reflection and would invert handedness.
        out[..., 0] = HALF_COURT_X - xy[..., 0]
        out[..., 1] = COURT_WIDTH - xy[..., 1]
    else:
        raise ValueError(f"basket must be {LEFT!r} or {RIGHT!r}, got {basket!r}")
    return out


def fold_vector(vec, basket: str):
    """Fold a *displacement* -- velocity, offset -- rather than a position.

    Only the linear part of the transform applies: the right half is unchanged and
    the left half is a 180 degree rotation, which negates both components.
    """
    vec = np.asarray(vec, dtype=float)
    if basket == RIGHT:
        return vec.copy()
    if basket == LEFT:
        return -vec
    raise ValueError(f"basket must be {LEFT!r} or {RIGHT!r}, got {basket!r}")


def unfold(xy, basket: str):
    """Inverse of :func:`fold`, for rendering predictions back on a full court."""
    xy = np.asarray(xy, dtype=float)
    out = np.empty_like(xy)
    if basket == RIGHT:
        out[..., 0] = xy[..., 0] + HALF_COURT_X
        out[..., 1] = xy[..., 1]
    elif basket == LEFT:
        out[..., 0] = HALF_COURT_X - xy[..., 0]
        out[..., 1] = COURT_WIDTH - xy[..., 1]
    else:
        raise ValueError(f"basket must be {LEFT!r} or {RIGHT!r}, got {basket!r}")
    return out


def rim_distance(xy):
    """Euclidean distance from the attacking rim, for already-folded coordinates."""
    xy = np.asarray(xy, dtype=float)
    return np.hypot(xy[..., 0] - HOOP[0], xy[..., 1] - HOOP[1])


def rim_angle(xy):
    """Signed bearing from the rim, in radians, for already-folded coordinates.

    Zero is straight out from the basket toward half court. The sign separates the
    two sides of the floor, which is the point of getting the fold right; see the
    module docstring on which physical side is which.

    The original used ``np.arctan2(diff[:, 0], diff[:, 1])`` -- arguments swapped
    relative to convention, so the angle was measured off the sideline axis. It was
    self-consistent because it was only ever used inside cosine differences, but the
    sign carried no meaning while the fold was broken.
    """
    xy = np.asarray(xy, dtype=float)
    dx = xy[..., 0] - HOOP[0]
    dy = xy[..., 1] - HOOP[1]
    # -dx so that the "out from the basket" direction is the zero bearing.
    return np.arctan2(dy, -dx)


def describe_side_convention(xy) -> str:
    """Label a folded position as ``"low-y"``, ``"high-y"`` or ``"middle"``.

    Helper for the empirical check that resolves which physical side of the floor
    low ``y`` corresponds to: run a known left-corner specialist's makes through
    this and see which bucket they fall in.
    """
    y = float(np.asarray(xy, dtype=float)[..., 1])
    if y < CENTER_Y - 5:
        return "low-y"
    if y > CENTER_Y + 5:
        return "high-y"
    return "middle"
