import math

import pytest
from robot_sequencer.geometry import Point
from robot_sequencer.plan import MAP_RESOLUTION, extract_vertices

GOAL = Point(2.0, 2.0)  # p_g of V-04

# Raw vertices of the Path of Theta* on M (robot-docs nota 0003), the same hand-written values
# as the test of check_v04: the centre of the start cell, a one-cell diagonal facet at the corner
# of the wall, and the goal reached through a neighbour cell.
NOTE_0003_RAW = [
    Point(-0.025, -0.025),
    Point(0.275, 1.675),
    Point(0.325, 1.725),
    Point(0.525, 1.825),
    Point(0.675, 1.825),
    Point(1.925, 1.925),
    Point(1.975, 1.975),
]
# Written by hand, not computed: segments 1.697, 0.292, 0.150 and 1.337 m (nota 0003). The facet
# is absorbed (deviation 0.017 m <= res/2) and the neighbour of the goal cell is dropped.
NOTE_0003_VERTICES = [Point(0.275, 1.675), Point(0.525, 1.825), Point(0.675, 1.825), GOAL]


def densify(raw: list[Point]) -> list[Point]:
    """Return the raw vertices with points every res between them, as Theta* interpolates."""
    points = [raw[0]]
    for start, end in zip(raw, raw[1:], strict=False):
        count = max(1, math.ceil(start.distance_to(end) / MAP_RESOLUTION))
        points += [
            Point(
                start.x + (end.x - start.x) * k / count,
                start.y + (end.y - start.y) * k / count,
            )
            for k in range(1, count + 1)
        ]
    return points


def assert_points_close(points: list[Point], expected: list[Point]) -> None:
    assert len(points) == len(expected), points
    for point, other in zip(points, expected, strict=True):
        assert point.distance_to(other) == pytest.approx(0.0, abs=1e-9), points


def test_note_0003_plan_gives_its_four_vertices():
    assert_points_close(extract_vertices(densify(NOTE_0003_RAW), GOAL), NOTE_0003_VERTICES)


def test_neighbour_of_the_goal_cell_is_dropped():
    raw = [Point(-0.025, -0.025), Point(1.025, -0.025), Point(1.925, 1.925), Point(1.975, 1.975)]
    assert_points_close(extract_vertices(densify(raw), GOAL), [Point(1.025, -0.025), GOAL])


def test_vertex_away_from_the_goal_cell_is_kept():
    raw = [Point(-0.025, -0.025), Point(1.975, -0.025), Point(1.975, 1.975)]
    assert_points_close(extract_vertices(densify(raw), GOAL), [Point(1.975, -0.025), GOAL])


def test_short_segment_far_from_the_new_segment_is_kept():
    # A 0.08 m step sideways: removing either end deviates 0.08 m, above res/2.
    raw = [
        Point(-0.025, -0.025),
        Point(1.0, 0.0),
        Point(1.0, 0.08),
        Point(2.0, 0.08),
        Point(1.975, 1.975),
    ]
    expected = [Point(1.0, 0.0), Point(1.0, 0.08), Point(2.0, 0.08), GOAL]
    assert_points_close(extract_vertices(densify(raw), GOAL), expected)


def test_straight_path_gives_only_the_goal():
    raw = [Point(-0.025, -0.025), Point(1.975, 1.975)]
    assert_points_close(extract_vertices(densify(raw), GOAL), [GOAL])


def test_path_of_one_cell_gives_only_the_goal():
    assert_points_close(extract_vertices([Point(1.975, 1.975)], GOAL), [GOAL])


def test_empty_path_is_a_failure_of_the_plan():
    with pytest.raises(ValueError):
        extract_vertices([], GOAL)
