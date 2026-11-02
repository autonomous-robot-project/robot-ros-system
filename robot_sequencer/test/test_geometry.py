import math

import pytest
from robot_sequencer.geometry import Point, Pose, wrap_angle


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


def test_pose_of_odom_in_map_converts_a_vertex_in_map_to_odom():
    # odom is 1 m along x of map, turned +90 deg: its x axis is y of map, so map (1, 1) is
    # odom (1, 0).
    odom_in_map = Pose(Point(1.0, 0.0), math.pi / 2)
    vertex = odom_in_map.convert_to_local(Point(1.0, 1.0))
    assert (vertex.x, vertex.y) == pytest.approx((1.0, 0.0))


def test_identity_pose_keeps_the_point():
    assert Pose(Point(0.0, 0.0), 0.0).convert_to_local(Point(2.0, 2.0)) == Point(2.0, 2.0)
