import math

import pytest
from robot_sequencer.geometry import Point, wrap_angle


@pytest.mark.parametrize(
    ('angle', 'wrapped'),
    [
        (0.0, 0.0),
        (math.pi / 2, math.pi / 2),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3 * math.pi / 2, -math.pi / 2),
        (-3 * math.pi / 2, math.pi / 2),
        (5 * math.pi, -math.pi),
    ],
)
def test_wrap_angle_returns_the_same_angle_in_minus_pi_to_pi(angle, wrapped):
    assert wrap_angle(angle) == pytest.approx(wrapped)


def test_distance_to_is_the_euclidean_distance():
    assert Point(1.0, 1.0).distance_to(Point(4.0, 5.0)) == pytest.approx(5.0)


@pytest.mark.parametrize(
    ('target', 'direction'),
    [
        (Point(1.0, 0.0), 0.0),
        (Point(0.0, 1.0), math.pi / 2),
        (Point(-1.0, 0.0), math.pi),
        (Point(0.0, -1.0), -math.pi / 2),
    ],
)
def test_direction_to_is_the_heading_of_the_line_to_the_target(target, direction):
    assert Point(0.0, 0.0).direction_to(target) == pytest.approx(direction)
