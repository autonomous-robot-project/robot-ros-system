"""
V-04: integration of the layers (RF-06, RF-07), judged on the ros2 bag recorded by run_v04.

Criticality: T1, evidence (robot-docs codigo.md).

What: the plan asked for is delivered; for its n segments the controller runs n pairs
TURN_TO -> GO_TO in order, with the vertices of the plan as the targets of the GO_TO, then
TWIST(0, 0); the final pose of the odometry is within ε_g of p_g.
How: the expected vertices are computed here from the recorded Path (/plan), with the rule of
ADR 034 written again from its text, never imported from the sequencer, which would judge itself
(F7 D5): drop the first point, replace the last by p_g and drop its neighbour vertex, absorb a
segment shorter than 2·ε_g when the removed vertex stays within res/2 of the new segment. The
echo of the State, with consecutive equal behaviours merged, must read TWIST(0, 0), the pairs,
TWIST(0, 0).
The heading of each TURN_TO is computed by the sequencer from the pose and is not judged.
Why: V-04 (robot-docs verifications/V-04).

Criterion (V-04): RF-06, the plan asked for is delivered to the sequencer; RF-07, for n segments of
the polyline of the plan the controller activates n pairs TURN_TO -> GO_TO, in the order of the
segments, with the vertices of the plan as the target of the GO_TO in the echo of the State,
followed by TWIST(0, 0), and the final pose by the odometry is within ε_g of p_g.

Validity of the run: one plan in the recording (one request); map -> odom is the identity in
/tf_static, as in the trial launch, so that the plan in map and the targets in odom compare
directly; the plan has a turn (n >= 2) and every segment is longer than 2·ε_g, so that a skipped or
swapped segment fails; the odometry is recorded after the final TWIST(0, 0) starts.

Usage: check_v04 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import (
    ControllerRecording,
    Sample,
    Stream,
    Verdict,
    read_bag,
)

if TYPE_CHECKING:
    from geometry_msgs.msg import Transform
    from nav_msgs.msg import Path
    from robot_interfaces.msg import Command
    from tf2_msgs.msg import TFMessage

PLAN_TOPIC = '/plan'
TF_STATIC_TOPIC = '/tf_static'

# Parameters of V-04.
P_0: tuple[float, float] = (0.0, 0.0)  # initial pose, the origin
P_G: tuple[float, float] = (2.0, 2.0)  # goal
EPSILON_G: float = 0.05  # m, ε_g = ε_d, the termination tolerance of the GO_TO
RESOLUTION: float = 0.05  # m, res of M
MIN_SEGMENTS: int = 2  # the plan has a turn

# Rule of ADR 034.
# Consecutive directions of the Path closer than this are collinear: with w_traversal_cost = 0 the
# vertices are the same from 1e-9 to 0.1 rad (robot-docs nota 0003).
COLLINEAR_TOLERANCE_RAD: float = 1e-3
# A vertex in a cell next to the goal cell: one cell away, also diagonally.
NEIGHBOUR_DISTANCE: float = RESOLUTION * math.sqrt(2) + 1e-6
MAX_ABSORPTION_DEVIATION: float = RESOLUTION / 2  # res/2

# Comparison of a target in the echo (float32, in odom) with a vertex (in map): well below res and
# far above the float32 rounding at 2 m (2.4e-7 m).
TARGET_TOLERANCE: float = 1e-3  # m
# map -> odom must be the identity; the static transform is written as exact zeros and one.
IDENTITY_TOLERANCE: float = 1e-9

# Values of the messages, checked against them when a bag is read.
TWIST: int = 0  # Command.TWIST
TURN_TO: int = 1  # Command.TURN_TO
GO_TO: int = 2  # Command.GO_TO

Point = tuple[float, float]


# --- Rule of ADR 034 ---


def find_direction_changes(points: list[Point]) -> list[Point]:
    """Return the first point, every point where the direction changes, and the last point."""
    changes = [
        current
        for previous, current, following in zip(points, points[1:], points[2:], strict=False)
        if not is_collinear(previous, current, following)
    ]
    return [points[0], *changes, points[-1]]


def is_collinear(previous: Point, current: Point, following: Point) -> bool:
    before = math.atan2(current[1] - previous[1], current[0] - previous[0])
    after = math.atan2(following[1] - current[1], following[0] - current[0])
    turn = math.remainder(after - before, math.tau)
    return abs(turn) <= COLLINEAR_TOLERANCE_RAD


def replace_ends(raw_vertices: list[Point], goal: Point) -> list[Point]:
    """
    Drop the first point and replace the last by the goal, dropping its neighbour vertex.

    The first point is the centre of the start cell, and the first segment starts at the pose.
    The last is the centre of the goal cell, which Theta* reaches through a neighbour cell.
    """
    goal_cell = raw_vertices[-1]
    interior = raw_vertices[1:-1]
    if interior and math.dist(interior[-1], goal_cell) <= NEIGHBOUR_DISTANCE:
        interior = interior[:-1]
    return [*interior, goal]


def absorb_short_segments(start: Point, vertices: list[Point]) -> list[Point]:
    """Remove a vertex of each segment shorter than 2·ε_g, if it stays within res/2 (ADR 034)."""
    absorbed = list(vertices)
    # Each pass removes one vertex, so at most len(vertices) passes.
    for _ in range(len(vertices)):
        removable = find_absorbable_vertex(start, absorbed)
        if removable is None:
            break
        absorbed.pop(removable)
    return absorbed


def find_absorbable_vertex(start: Point, vertices: list[Point]) -> int | None:
    """Return the position in vertices of the vertex to remove for a short segment, or None."""
    polyline = [start, *vertices]
    short_ends = [end for end in range(1, len(polyline)) if is_short(polyline, end)]
    for end in short_ends:
        chosen = choose_vertex_to_absorb(polyline, end)
        if chosen is not None:
            return chosen - 1  # polyline starts with the start, vertices does not
    return None


def is_short(polyline: list[Point], end: int) -> bool:
    """Tell whether the segment that ends at polyline[end] is shorter than 2·ε_g."""
    return math.dist(polyline[end - 1], polyline[end]) < 2 * EPSILON_G


def choose_vertex_to_absorb(polyline: list[Point], end: int) -> int | None:
    """
    Return the end of a short segment to remove, or None.

    The candidates are its ends other than the start and the goal; the one that deviates least
    wins, if it stays within res/2 of the segment that replaces it.
    """
    candidates = [c for c in (end - 1, end) if 0 < c < len(polyline) - 1]
    allowed = [c for c in candidates if measure_deviation(polyline, c) <= MAX_ABSORPTION_DEVIATION]
    return min(allowed, key=lambda c: measure_deviation(polyline, c), default=None)


def measure_deviation(polyline: list[Point], position: int) -> float:
    """Return the distance of a vertex from the segment between its neighbours."""
    return distance_to_line(polyline[position], polyline[position - 1], polyline[position + 1])


def distance_to_line(point: Point, start: Point, end: Point) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    cross = dx * (point[1] - start[1]) - dy * (point[0] - start[0])
    return abs(cross) / math.hypot(dx, dy)


def extract_vertices(path_points: list[Point]) -> list[Point]:
    """Return the vertices of the plan, the targets of the GO_TO, by the rule of ADR 034."""
    raw = find_direction_changes(path_points)
    return absorb_short_segments(P_0, replace_ends(raw, P_G))


# --- The run ---


@dataclass(frozen=True)
class EchoedBehavior:
    """
    A behaviour in force in the echo, over consecutive States; the target is that of a GO_TO.

    One per State, then merged, so the echo reads as the order of the behaviours (RF-07).
    """

    behavior: int
    target: Point
    is_stop: bool  # TWIST(0, 0)


def to_echoed_behavior(echo: Command) -> EchoedBehavior:
    is_stop = echo.behavior == TWIST and echo.linear == 0.0 and echo.angular == 0.0
    return EchoedBehavior(echo.behavior, (echo.x, echo.y), is_stop)


def merge_repeated_behaviors(behaviors: list[EchoedBehavior]) -> list[EchoedBehavior]:
    """Merge consecutive equal behaviours; a GO_TO with a new target stays apart."""
    return [behavior for behavior, _ in itertools.groupby(behaviors)]


def is_identity_map_to_odom(transforms: TFMessage) -> bool:
    return any(
        t.header.frame_id == 'map' and t.child_frame_id == 'odom' and is_identity(t.transform)
        for t in transforms.transforms
    )


def is_identity(transform: Transform) -> bool:
    values = [
        transform.translation.x,
        transform.translation.y,
        transform.translation.z,
        transform.rotation.x,
        transform.rotation.y,
        transform.rotation.z,
        transform.rotation.w - 1.0,
    ]
    return all(abs(value) <= IDENTITY_TOLERANCE for value in values)


def to_points(path: Path) -> list[Point]:
    return [(pose.pose.position.x, pose.pose.position.y) for pose in path.poses]


class PlanRun:
    """The plan, the static transforms and the controller topics of a recording."""

    def __init__(
        self,
        recording: ControllerRecording,
        plans: Stream[Path],
        static_transforms: Stream[TFMessage],
    ):
        self.recording: ControllerRecording = recording
        self.plans: Stream[Path] = plans
        self.static_transforms: Stream[TFMessage] = static_transforms

    @classmethod
    def from_bag(cls, path: str) -> PlanRun:
        # ROS imports here, so the criterion can be tested without ROS.
        from nav_msgs.msg import Path
        from robot_interfaces.msg import Command
        from tf2_msgs.msg import TFMessage

        assert (Command.TWIST, Command.TURN_TO, Command.GO_TO) == (TWIST, TURN_TO, GO_TO)
        streams = read_bag(path, {PLAN_TOPIC: Path, TF_STATIC_TOPIC: TFMessage})
        return cls(
            ControllerRecording.from_bag(path), streams[PLAN_TOPIC], streams[TF_STATIC_TOPIC]
        )

    def get_expected_vertices(self) -> list[Point]:
        """Return the vertices of the first plan; empty without a plan of two points."""
        if not self.plans or len(self.plans[0].message.poses) < 2:
            return []
        return extract_vertices(to_points(self.plans[0].message))

    def get_echoed_behaviors(self) -> list[EchoedBehavior]:
        echoed = [to_echoed_behavior(s.message.echo) for s in self.recording.states]
        return merge_repeated_behaviors(echoed)

    def get_final_position(self) -> Point | None:
        if not self.recording.odometry:
            return None
        position = self.recording.odometry[-1].message.pose.pose.position
        return (position.x, position.y)

    def find_final_stop_start_s(self) -> float | None:
        """Return when the last run of TWIST(0, 0) in the echo started, or None."""
        start_s = None
        for previous, current in itertools.pairwise(self.recording.states):
            if not is_stop_sample(previous) and is_stop_sample(current):
                start_s = current.time_s
        return start_s

    def has_identity_map_to_odom(self) -> bool:
        return any(is_identity_map_to_odom(s.message) for s in self.static_transforms)


def is_stop_sample(sample: Sample) -> bool:
    return to_echoed_behavior(sample.message.echo).is_stop


def build_expected_behaviors(vertices: list[Point]) -> list[tuple[int, Point | None]]:
    """Return the expected (behaviour, target) after the merge: TWIST(0, 0), pairs, TWIST(0, 0)."""
    pairs = [pair for vertex in vertices for pair in ((TURN_TO, None), (GO_TO, vertex))]
    return [(TWIST, None), *pairs, (TWIST, None)]


def does_behavior_match(echoed: EchoedBehavior, expected: tuple[int, Point | None]) -> bool:
    expected_behavior, expected_target = expected
    if echoed.behavior != expected_behavior:
        return False
    if expected_behavior == TWIST:
        return echoed.is_stop
    if expected_behavior == GO_TO:
        return math.dist(echoed.target, expected_target) <= TARGET_TOLERANCE
    return True


class V04:
    """
    The criterion and the validity conditions of V-04, and its verdict.

    Valid run: one plan, map -> odom as the identity, a plan with a turn and segments longer than
    2·ε_g, and the odometry recorded after the final stop.
    """

    def __init__(self, run: PlanRun):
        self.run: PlanRun = run

    # --- Verdict ---

    @property
    def verdict(self) -> Verdict:
        if not self.was_plan_delivered():
            return Verdict.FAIL
        if not self.run_is_valid:
            return Verdict.INVALID
        passed = self.did_echo_follow_the_plan() and self.did_end_within_epsilon_g()
        return Verdict.PASS if passed else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        return (
            self.has_one_plan()
            and self.run.has_identity_map_to_odom()
            and self.has_a_turn()
            and self.are_segments_longer_than_two_epsilon_g()
            and self.was_odometry_recorded_after_the_stop()
        )

    # --- Criterion ---

    def was_plan_delivered(self) -> bool:
        """RF-06: a plan of at least two points is in the recording."""
        return bool(self.run.get_expected_vertices())

    def did_echo_follow_the_plan(self) -> bool:
        """RF-07: TWIST(0, 0), one pair TURN_TO -> GO_TO(vertex) per segment, TWIST(0, 0)."""
        echoed = self.run.get_echoed_behaviors()
        expected = build_expected_behaviors(self.run.get_expected_vertices())
        if len(echoed) != len(expected):
            return False
        return all(does_behavior_match(b, e) for b, e in zip(echoed, expected, strict=True))

    def did_end_within_epsilon_g(self) -> bool:
        return self.measure_final_distance() <= EPSILON_G

    def measure_final_distance(self) -> float:
        final = self.run.get_final_position()
        return math.inf if final is None else math.dist(final, P_G)

    # --- Validity of the run ---

    def has_one_plan(self) -> bool:
        return len(self.run.plans) == 1

    def has_a_turn(self) -> bool:
        return len(self.run.get_expected_vertices()) >= MIN_SEGMENTS

    def are_segments_longer_than_two_epsilon_g(self) -> bool:
        polyline = [P_0, *self.run.get_expected_vertices()]
        return all(math.dist(a, b) > 2 * EPSILON_G for a, b in itertools.pairwise(polyline))

    def was_odometry_recorded_after_the_stop(self) -> bool:
        stop_s = self.run.find_final_stop_start_s()
        odometry = self.run.recording.odometry
        return stop_s is not None and bool(odometry) and odometry[-1].time_s > stop_s


class V04Report:
    """The report of V-04: the plan, the echo and the verdict of each check."""

    def __init__(self, verification: V04):
        self.verification: V04 = verification
        self.run: PlanRun = verification.run

    def __str__(self) -> str:
        v = self.verification
        vertices = self.run.get_expected_vertices()
        lines = [
            f'V-04: {len(self.run.plans)} plans, {len(vertices)} segments expected',
            f'  vertices {format_points(vertices)}',
            f'  echo {format_behaviors(self.run.get_echoed_behaviors())}',
            f'  RF-06 plan delivered, passed: {v.was_plan_delivered()}',
            f'  RF-07 pairs TURN_TO -> GO_TO in order, passed: {v.did_echo_follow_the_plan()}',
            f'  final distance to p_g {v.measure_final_distance():.3f} m (max {EPSILON_G} m), '
            f'passed: {v.did_end_within_epsilon_g()}',
            *self._format_validity_lines(),
            f'V-04: {v.verdict.name}',
        ]
        return '\n'.join(lines)

    def _format_validity_lines(self) -> list[str]:
        v = self.verification
        return [
            f'  one plan, passed: {v.has_one_plan()}',
            f'  map -> odom identity, passed: {self.run.has_identity_map_to_odom()}',
            f'  at least {MIN_SEGMENTS} segments, passed: {v.has_a_turn()}',
            f'  segments longer than 2·ε_g, passed: {v.are_segments_longer_than_two_epsilon_g()}',
            f'  odometry after the final stop, passed: {v.was_odometry_recorded_after_the_stop()}',
        ]


def format_points(points: list[Point]) -> str:
    return ' '.join(f'({x:.3f}, {y:.3f})' for x, y in points)


def format_behaviors(behaviors: list[EchoedBehavior]) -> str:
    names = {TWIST: 'TWIST', TURN_TO: 'TURN_TO', GO_TO: 'GO_TO'}
    return ' -> '.join(names.get(b.behavior, str(b.behavior)) for b in behaviors)


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v04 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    verification = V04(PlanRun.from_bag(sys.argv[1]))
    print(V04Report(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
