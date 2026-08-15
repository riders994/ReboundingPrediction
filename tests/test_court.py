"""Tests for half-court folding.

The central property: the *same play run at the other end* must fold to the same
half-court coordinates. A play at the left basket is the 180 degree rotation about
center court of the equivalent play at the right basket, since that is the symmetry
the court actually has.
"""

import numpy as np
import pytest

from rebounding.constants import COURT_LENGTH, COURT_WIDTH, HOOP, RIM_LEFT, RIM_RIGHT
from rebounding.data.court import (
    LEFT,
    RIGHT,
    attacking_basket,
    fold,
    rim_angle,
    rim_distance,
    unfold,
)


def rotate_to_other_end(xy):
    """The same play, run at the opposite basket."""
    xy = np.asarray(xy, dtype=float)
    return np.stack([COURT_LENGTH - xy[..., 0], COURT_WIDTH - xy[..., 1]], axis=-1)


# A ten-player configuration around the right basket, deliberately asymmetric in y
# and including one trailing defender who has not crossed half court.
RIGHT_END_PLAY = np.array(
    [
        [88.0, 24.0],  # under the rim
        [84.0, 10.0],  # low-y wing
        [80.0, 40.0],  # high-y wing
        [89.0, 3.0],   # low-y corner
        [70.0, 25.0],  # top of the key
        [86.0, 20.0],
        [83.0, 30.0],
        [87.0, 12.0],
        [75.0, 18.0],
        [40.0, 25.0],  # trailing defender, still in the back court
    ]
)


def test_both_rims_fold_to_the_same_hoop():
    assert fold(RIM_RIGHT, RIGHT) == pytest.approx(HOOP)
    assert fold(RIM_LEFT, LEFT) == pytest.approx(HOOP)


def test_same_play_at_either_end_folds_identically():
    """The property the old abs(x - 47) fold violated."""
    folded_right = fold(RIGHT_END_PLAY, RIGHT)
    folded_left = fold(rotate_to_other_end(RIGHT_END_PLAY), LEFT)
    np.testing.assert_allclose(folded_right, folded_left)


def test_old_reflection_fold_mirrors_the_other_end():
    """Demonstrates the bug being fixed, so the test suite documents it.

    ``abs(x - 47)`` with y untouched sends the same play at the two ends to
    y-mirrored positions, superimposing left-wing and right-wing shots.
    """
    def old_fold(xy):
        xy = np.asarray(xy, dtype=float)
        return np.stack([np.abs(xy[..., 0] - 47.0), xy[..., 1]], axis=-1)

    old_right = old_fold(RIGHT_END_PLAY)
    old_left = old_fold(rotate_to_other_end(RIGHT_END_PLAY))

    assert not np.allclose(old_right, old_left)
    # Specifically, y comes out mirrored about the center line.
    np.testing.assert_allclose(old_left[:, 1], COURT_WIDTH - old_right[:, 1])


def test_sidedness_is_preserved_across_ends():
    """A low-y shot at one end must stay low-y after folding at either end."""
    low_y_corner = np.array([89.0, 3.0])
    folded_right = fold(low_y_corner, RIGHT)
    folded_left = fold(rotate_to_other_end(low_y_corner), LEFT)
    assert folded_right[1] < 25.0
    assert folded_left[1] < 25.0


def test_backcourt_player_keeps_true_rim_distance():
    """The per-player abs() fold reflected back-court players into the front court."""
    trailer = np.array([40.0, 25.0])
    true_distance = np.hypot(RIM_RIGHT[0] - 40.0, RIM_RIGHT[1] - 25.0)  # 48.75

    folded = fold(trailer, RIGHT)
    assert rim_distance(folded) == pytest.approx(true_distance)

    # What the old code produced: abs(40 - 47) = 7, i.e. 34.75 ft from the rim.
    old_folded = np.array([abs(40.0 - 47.0), 25.0])
    assert rim_distance(old_folded) == pytest.approx(34.75)


def test_fold_unfold_roundtrip():
    for basket, play in ((RIGHT, RIGHT_END_PLAY), (LEFT, rotate_to_other_end(RIGHT_END_PLAY))):
        np.testing.assert_allclose(unfold(fold(play, basket), basket), play)


def test_attacking_basket_from_ball():
    assert attacking_basket([88.6, 25.2]) == RIGHT
    assert attacking_basket([5.4, 24.8]) == LEFT


def test_rim_angle_is_signed_and_zero_straight_on():
    straight_on = np.array([HOOP[0] - 20.0, HOOP[1]])
    assert rim_angle(straight_on) == pytest.approx(0.0)

    high_y = np.array([HOOP[0] - 20.0, HOOP[1] + 10.0])
    low_y = np.array([HOOP[0] - 20.0, HOOP[1] - 10.0])
    assert rim_angle(high_y) > 0
    assert rim_angle(low_y) < 0
    assert rim_angle(high_y) == pytest.approx(-rim_angle(low_y))


def test_fold_rejects_bad_basket_and_shape():
    with pytest.raises(ValueError):
        fold(RIGHT_END_PLAY, "middle")
    with pytest.raises(ValueError):
        fold(np.zeros((10, 3)), RIGHT)
