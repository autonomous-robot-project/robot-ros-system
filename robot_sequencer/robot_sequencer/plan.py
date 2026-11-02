"""
Vertices of the plan: the polyline of segments that the Path of Theta* is made of (ADR 034).

Criticality: T2, product (robot-docs codigo.md).

Theta* chooses its vertices by line of sight and interpolates between them at the resolution of
the map, so the vertices are the points where the direction changes. The ends and the short
segments are then handled as ADR 034 says:
- the first point, the centre of the start cell, is dropped: the first segment leaves from the
  current pose;
- the last point, the centre of the goal cell, becomes the goal asked for, and the vertex next to
  that cell, if any, is dropped;
- a segment shorter than 2·ε_g loses one of its inner vertices, if that vertex lies within res/2
  of the new segment; the cell added to the forbidden zone covers this deviation (ADR 042).
A linear pass, with no search (RNF-03). Domain without ROS (ADR 047): the node converts the Path.
"""

from collections.abc import Sequence

from robot_sequencer.geometry import Point, wrap_angle
from robot_sequencer.mission import EPSILON_D

MAP_RESOLUTION: float = 0.05  # m, res of the map and of the costmap (ADR 041)
EPSILON_G: float = EPSILON_D  # m, ε_g: the goal is reached by the termination of GO_TO (V-04)
SHORT_SEGMENT: float = 2 * EPSILON_G  # m, below it a segment is absorbed (ADR 034)
MAX_ABSORBED_DEVIATION: float = MAP_RESOLUTION / 2  # m, res/2 (ADR 034)
# The interpolated points are collinear: any value from 1e-9 to 0.1 rad gives the same vertices
# on a real Path (robot-docs nota 0003).
DIRECTION_TOLERANCE: float = 0.01  # rad
# The vertex next to the goal cell is one cell away, straight (res) or diagonal (√2·res), never
# two cells (2·res) (nota 0003): the bound sits between them.
GOAL_NEIGHBOUR_DISTANCE: float = 1.5 * MAP_RESOLUTION  # m


def get_distance_to_segment(point: Point, start: Point, end: Point) -> float:
    length = start.distance_to(end)
    if length == 0.0:
        return point.distance_to(start)
    # Projection of the point on the segment, clamped to its ends.
    along = (
        (point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y)
    ) / length**2
    along = min(1.0, max(0.0, along))
    closest = Point(start.x + along * (end.x - start.x), start.y + along * (end.y - start.y))
    return point.distance_to(closest)


def get_removable_vertex(polyline: list[Point], first: int, second: int) -> int | None:
    """Return the inner end of the segment that deviates least once removed, if within res/2."""
    inner_ends = [index for index in (first, second) if 0 < index < len(polyline) - 1]
    deviations = {
        index: get_distance_to_segment(polyline[index], polyline[index - 1], polyline[index + 1])
        for index in inner_ends
    }
    if not deviations:
        return None
    index = min(deviations, key=deviations.__getitem__)
    return index if deviations[index] <= MAX_ABSORBED_DEVIATION else None


def absorb_short_segments(start: Point, vertices: list[Point]) -> list[Point]:
    """
    Return the polyline from start with each short segment absorbed where the deviation allows.

    Of the two ends of a short segment, the inner one with the smaller deviation is removed; the
    start and the goal stay. The segment that follows is tested again against its new neighbour.
    """
    polyline = [start, *vertices]
    index = 1
    while index < len(polyline):
        if polyline[index - 1].distance_to(polyline[index]) < SHORT_SEGMENT:
            removable = get_removable_vertex(polyline, index - 1, index)
            if removable is not None:
                del polyline[removable]
                index = max(1, removable)
                continue
        index += 1
    return polyline


def find_direction_changes(path: Sequence[Point]) -> list[Point]:
    """Return the first point, each point where the direction changes, and the last point."""
    turns = [path[0]]
    for before, point, after in zip(path, path[1:], path[2:], strict=False):
        change = wrap_angle(point.direction_to(after) - before.direction_to(point))
        if abs(change) > DIRECTION_TOLERANCE:
            turns.append(point)
    if len(path) > 1:
        turns.append(path[-1])
    return turns


def extract_vertices(path: Sequence[Point], goal: Point) -> list[Point]:
    """Return the vertices in map of the segments from the current pose to the goal."""
    if not path:
        raise ValueError('an empty path has no vertices: a failure of the plan (ADR 034)')
    turns = find_direction_changes(path)
    start_cell, inner, goal_cell = turns[0], turns[1:-1], turns[-1]
    if inner and inner[-1].distance_to(goal_cell) <= GOAL_NEIGHBOUR_DISTANCE:
        inner.pop()
    return absorb_short_segments(start_cell, [*inner, goal])[1:]
