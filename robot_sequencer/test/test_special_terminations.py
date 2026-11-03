"""The special terminations of ADR 036: a turn above θ_max in parts, and NO_PROGRESS."""

import math

import pytest
from robot_sequencer.geometry import Point, Pose, wrap_angle
from robot_sequencer.mission import (
    STOP,
    THETA_MAX,
    Behavior,
    Command,
    Condition,
    Ignore,
    Mission,
    OdometryReading,
    limit_turn,
)
from robot_sequencer.missions import build_plan_mission, build_v01_mission
from robot_sequencer.sequencer import STALL_READINGS, MissionStatus, Sequencer

IDENTITY = Pose(Point(0.0, 0.0), 0.0)
BEHIND = Point(-1.0, 0.01)  # 179.4 deg from the heading 0 at the origin, to the left


def at_rest(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.0, angular=0.0)


def moving(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.2, angular=0.0)


def build_plan_sequencer(mission: Mission) -> Sequencer:
    return Sequencer(mission, get_map_to_odom=lambda: IDENTITY)


def build_moved_go_to() -> Sequencer:
    """Return a sequencer in the GO_TO of a plan to (1, 0), after one reading in motion."""
    sequencer = build_plan_sequencer(build_plan_mission([Point(1.0, 0.0)]))
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # TURN_TO, already aligned
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # GO_TO
    sequencer.on_odometry(moving(0.5, 0.0, 0.0))
    return sequencer


def stay_at_rest(sequencer: Sequencer, readings: int) -> list[Command | None]:
    return [sequencer.on_odometry(at_rest(0.8, 0.0, 0.0)) for _ in range(readings)]


@pytest.mark.parametrize('theta', [0.0, math.radians(90.0), THETA_MAX, -THETA_MAX])
def test_turn_up_to_theta_max_is_kept(theta):
    command = Command(Behavior.TURN_TO, theta=theta)
    assert limit_turn(command, IDENTITY) == command


@pytest.mark.parametrize('sign', [1.0, -1.0])
def test_turn_above_theta_max_is_cut_to_theta_max_toward_the_target(sign):
    pose = Pose(Point(0.0, 0.0), math.radians(10.0))
    command = Command(Behavior.TURN_TO, theta=math.radians(10.0 + sign * 175.0))
    limited = limit_turn(command, pose)
    expected = math.radians(10.0) + sign * THETA_MAX
    assert wrap_angle(limited.theta - expected) == pytest.approx(0.0)


def test_twist_is_not_limited():
    command = Command(Behavior.TWIST, angular=1.0)
    assert limit_turn(command, Pose(Point(0.0, 0.0), 3.0)) == command


def test_turn_to_a_target_behind_is_done_in_two_parts():
    sequencer = build_plan_sequencer(build_plan_mission([BEHIND]))
    heading_to_target = math.atan2(BEHIND.y, BEHIND.x)

    first = sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    assert first is not None and first.theta == pytest.approx(THETA_MAX)

    second = sequencer.on_odometry(at_rest(0.0, 0.0, THETA_MAX))  # the partial target reached
    assert second is not None and second.behavior is Behavior.TURN_TO
    assert second.theta == pytest.approx(heading_to_target)

    go = sequencer.on_odometry(at_rest(0.0, 0.0, heading_to_target))
    assert go is not None and go.behavior is Behavior.GO_TO


def test_go_to_at_rest_out_of_tolerance_for_t_stall_after_moving_aborts():
    sequencer = build_moved_go_to()

    assert stay_at_rest(sequencer, STALL_READINGS)[-1] == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_rest_shorter_than_t_stall_is_not_a_stall():
    # One edge and then T_vel at rest, while the ramp crosses the dead zone (robot-docs nota 0018).
    sequencer = build_moved_go_to()

    assert stay_at_rest(sequencer, STALL_READINGS - 1) == [None] * (STALL_READINGS - 1)
    sequencer.on_odometry(moving(0.9, 0.0, 0.0))  # the wheels get going
    assert stay_at_rest(sequencer, STALL_READINGS - 1) == [None] * (STALL_READINGS - 1)
    assert sequencer.status is MissionStatus.RUNNING


def test_rest_at_the_entry_of_the_step_is_not_a_stall():
    sequencer = build_plan_sequencer(build_plan_mission([Point(1.0, 0.0)]))
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # TURN_TO
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # GO_TO, not moved yet

    rests = [sequencer.on_odometry(at_rest(0.0, 0.0, 0.0)) for _ in range(2 * STALL_READINGS)]
    assert rests == [None] * (2 * STALL_READINGS)
    assert sequencer.status is MissionStatus.RUNNING


def test_twist_to_a_distance_has_no_tolerance_and_no_stall():
    sequencer = Sequencer(build_v01_mission())
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # S1
    sequencer.on_odometry(moving(0.5, 0.0, 0.0))

    rests = [sequencer.on_odometry(at_rest(0.5, 0.0, 0.0)) for _ in range(2 * STALL_READINGS)]
    assert rests == [None] * (2 * STALL_READINGS)
    assert sequencer.status is MissionStatus.RUNNING


def test_ignored_no_progress_is_raised_once_per_stop():
    plan = build_plan_mission([Point(1.0, 0.0)])
    mission = Mission(plan.steps, reactions={Condition.NO_PROGRESS: Ignore()})
    sequencer = build_plan_sequencer(mission)
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # TURN_TO
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # GO_TO
    sequencer.on_odometry(moving(0.5, 0.0, 0.0))

    assert stay_at_rest(sequencer, 2 * STALL_READINGS) == [None] * (2 * STALL_READINGS)
    assert sequencer.status is MissionStatus.RUNNING
