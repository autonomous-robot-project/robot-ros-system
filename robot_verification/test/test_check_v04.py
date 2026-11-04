"""Unit tests of the V-04 criterion and of the rule of ADR 034 on synthetic recordings."""

from dataclasses import dataclass, field
import math
from types import SimpleNamespace

import pytest
from robot_verification.check_v04 import (
    GO_TO,
    MAP_RESOLUTION,
    P_G,
    TURN_TO,
    TWIST,
    V04,
    PlanRun,
    V04Report,
    extract_vertices,
)
from robot_verification.recording import ControllerRecording, Sample, Stream, Verdict
from synthetic import odometry, state, to_float32

# Raw vertices of the Path of Theta* on M (robot-docs nota 0003): the centre of the start cell,
# a one-cell diagonal facet at the corner of the wall, and the goal reached through a
# neighbour cell.
NOTE_0003_RAW = [
    (-0.025, -0.025),
    (0.275, 1.675),
    (0.325, 1.725),
    (0.525, 1.825),
    (0.675, 1.825),
    (1.925, 1.925),
    (1.975, 1.975),
]
# Written by hand from nota 0003, not computed: segments 1.697, 0.292, 0.150 and 1.337 m. The
# facet is absorbed (deviation 0.017 m <= res/2) and the neighbour of the goal cell is dropped.
NOTE_0003_VERTICES = [(0.275, 1.675), (0.525, 1.825), (0.675, 1.825), (2.0, 2.0)]

STATE_PERIOD_S = 0.01  # 100 Hz
ODOMETRY_PERIOD_S = 0.05  # 20 Hz
IDLE_S = 1.0
TURN_S = 0.5
GO_S = 1.0
FINAL_STOP_S = 3.0
FINAL_OFFSET = 0.02  # m from p_g, within ε_g


def densify(raw: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the raw vertices with points every res between them, as Theta* interpolates."""
    points = [raw[0]]
    for start, end in zip(raw, raw[1:], strict=False):
        count = max(1, math.ceil(math.dist(start, end) / MAP_RESOLUTION))
        points += [
            (
                start[0] + (end[0] - start[0]) * k / count,
                start[1] + (end[1] - start[1]) * k / count,
            )
            for k in range(1, count + 1)
        ]
    return points


def path(points):
    poses = [
        SimpleNamespace(pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y)))
        for x, y in points
    ]
    return SimpleNamespace(poses=poses)


def map_to_odom(x=0.0):
    transform = SimpleNamespace(
        translation=SimpleNamespace(x=x, y=0.0, z=0.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    return SimpleNamespace(
        transforms=[
            SimpleNamespace(
                header=SimpleNamespace(frame_id='map'), child_frame_id='odom', transform=transform
            )
        ]
    )


def echo(behavior, x=0.0, y=0.0, theta=0.0):
    return SimpleNamespace(
        behavior=behavior, linear=0.0, angular=0.0, theta=theta, x=to_float32(x), y=to_float32(y)
    )


@dataclass
class SyntheticV04Run:
    """
    A V-04 recording from a sequencer and a controller that behave as configured.

    The echo is TWIST(0, 0), then TURN_TO and GO_TO(vertex) for each vertex, then TWIST(0, 0); the
    odometry ends at the final position.
    """

    raw_plan: list[tuple[float, float]] = field(default_factory=lambda: list(NOTE_0003_RAW))
    plans: int = 1
    map_to_odom_x: float = 0.0
    echoed_vertices: list[tuple[float, float]] = field(
        default_factory=lambda: list(NOTE_0003_VERTICES)
    )
    stop_before: int | None = None  # TWIST(0, 0) in the echo before this pair (an expiry)
    final_position: tuple[float, float] = (P_G[0] - FINAL_OFFSET, P_G[1])
    odometry_ends_early: bool = False

    def run(self) -> PlanRun:
        states = self._states()
        end_s = states[-1].time_s
        # Cut half a second before the final TWIST(0, 0) starts.
        cut_s = end_s - FINAL_STOP_S - 0.5
        odometry_end_s = cut_s if self.odometry_ends_early else end_s
        final_x, final_y = self.final_position
        odometry_samples = [
            Sample(k * ODOMETRY_PERIOD_S, odometry(x=final_x, y=final_y))
            for k in range(round(odometry_end_s / ODOMETRY_PERIOD_S) + 1)
        ]
        recording = ControllerRecording([], states, [], odometry_samples)
        plans = [Sample(0.5, path(densify(self.raw_plan))) for _ in range(self.plans)]
        transforms = [Sample(0.0, map_to_odom(self.map_to_odom_x))]
        return PlanRun(recording, Stream(plans), Stream(transforms))

    def _states(self) -> list[Sample]:
        echoes = [(echo(TWIST), IDLE_S)]
        for position, vertex in enumerate(self.echoed_vertices):
            if position == self.stop_before:
                echoes.append((echo(TWIST), IDLE_S))
            echoes += [(echo(TURN_TO, theta=1.0), TURN_S), (echo(GO_TO, *vertex), GO_S)]
        echoes.append((echo(TWIST), FINAL_STOP_S))
        return to_samples(echoes)


def to_samples(echoes) -> list[Sample]:
    samples, time_s = [], 0.0
    for message, duration_s in echoes:
        for _ in range(round(duration_s / STATE_PERIOD_S)):
            samples.append(Sample(time_s, state(message)))
            time_s += STATE_PERIOD_S
    return samples


def judge(run: SyntheticV04Run) -> V04:
    return V04(run.run())


def assert_points_close(points, expected):
    assert len(points) == len(expected)
    for point, reference in zip(points, expected, strict=True):
        assert math.dist(point, reference) < 1e-9


# --- Rule of ADR 034 ---


def test_note_0003_plan_gives_its_four_vertices():
    assert_points_close(extract_vertices(densify(NOTE_0003_RAW)), NOTE_0003_VERTICES)


def test_neighbour_of_the_goal_cell_is_dropped():
    raw = [(-0.025, -0.025), (1.025, -0.025), (1.925, 1.925), (1.975, 1.975)]
    assert_points_close(extract_vertices(densify(raw)), [(1.025, -0.025), (2.0, 2.0)])


def test_short_segment_far_from_the_new_segment_is_kept():
    # A 0.08 m step sideways: removing either end deviates 0.08 m, above res/2.
    raw = [(-0.025, -0.025), (1.0, 0.0), (1.0, 0.08), (2.0, 0.08), (1.975, 1.975)]
    expected = [(1.0, 0.0), (1.0, 0.08), (2.0, 0.08), P_G]
    assert_points_close(extract_vertices(densify(raw)), expected)


# --- Criterion ---


def test_echo_that_follows_the_plan_passes():
    assert judge(SyntheticV04Run()).verdict is Verdict.PASS


def test_without_a_plan_fails_rf_06():
    assert judge(SyntheticV04Run(plans=0)).verdict is Verdict.FAIL


@pytest.mark.parametrize(
    'echoed',
    [
        NOTE_0003_VERTICES[:1] + NOTE_0003_VERTICES[2:],  # a skipped segment
        [NOTE_0003_VERTICES[1], NOTE_0003_VERTICES[0], *NOTE_0003_VERTICES[2:]],  # swapped
        [*NOTE_0003_VERTICES[:3], (2.0, 1.95)],  # a wrong target
    ],
)
def test_echo_that_departs_from_the_plan_fails(echoed):
    assert judge(SyntheticV04Run(echoed_vertices=echoed)).verdict is Verdict.FAIL


def test_stop_in_the_middle_fails():
    assert judge(SyntheticV04Run(stop_before=2)).verdict is Verdict.FAIL


def test_final_pose_beyond_epsilon_g_fails():
    run = SyntheticV04Run(final_position=(P_G[0] - 0.06, P_G[1]))
    assert judge(run).verdict is Verdict.FAIL


# --- Validity ---


def test_two_plans_are_invalid():
    assert judge(SyntheticV04Run(plans=2)).verdict is Verdict.INVALID


def test_map_to_odom_not_identity_is_invalid():
    assert judge(SyntheticV04Run(map_to_odom_x=0.1)).verdict is Verdict.INVALID


def test_plan_without_a_turn_is_invalid():
    raw = [(-0.025, -0.025), (1.975, 1.975)]
    run = SyntheticV04Run(raw_plan=raw, echoed_vertices=[P_G])
    assert judge(run).verdict is Verdict.INVALID


def test_segment_shorter_than_two_epsilon_g_is_invalid():
    raw = [(-0.025, -0.025), (1.0, 0.0), (1.0, 0.08), (2.0, 0.08), (1.975, 1.975)]
    run = SyntheticV04Run(raw_plan=raw, echoed_vertices=extract_vertices(densify(raw)))
    assert judge(run).verdict is Verdict.INVALID


def test_odometry_cut_before_the_final_stop_is_invalid():
    assert judge(SyntheticV04Run(odometry_ends_early=True)).verdict is Verdict.INVALID


def test_report_lists_the_vertices_and_the_echo():
    report = str(V04Report(judge(SyntheticV04Run())))
    assert '(0.275, 1.675) (0.525, 1.825) (0.675, 1.825) (2.000, 2.000)' in report
    assert 'TWIST -> TURN_TO -> GO_TO' in report
    assert report.endswith('V-04: PASS')
