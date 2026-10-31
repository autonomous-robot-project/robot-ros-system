"""The engine on the mission of V-01: S1 TWIST, S2 TURN_TO, S3 GO_TO, S4 TWIST(0, 0)."""

import math

import pytest
from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import STOP, Behavior, Code, Command, OdometryReading
from robot_sequencer.missions import build_v01_mission
from robot_sequencer.sequencer import MissionStatus, Sequencer

NO_CODES: frozenset[Code] = frozenset()
FAULT = frozenset({Code.FAULT_COGNIZANT})


def moving(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.2, angular=0.0)


def at_rest(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.0, angular=0.0)


@pytest.fixture
def sequencer() -> Sequencer:
    return Sequencer(build_v01_mission())


def test_before_the_first_odometry_it_waits_with_twist_zero(sequencer):
    assert sequencer.on_state(FAULT) is None
    assert sequencer.status is MissionStatus.WAITING
    assert sequencer.command == STOP


def test_the_first_odometry_begins_the_first_step(sequencer):
    assert sequencer.on_odometry(at_rest(0.0, 0.0, 0.0)) == Command(Behavior.TWIST, linear=0.2)
    assert sequencer.status is MissionStatus.RUNNING


def test_nominal_run_goes_through_the_steps_and_succeeds(sequencer):
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    assert sequencer.on_odometry(moving(0.5, 0.0, 0.0)) is None

    turn = sequencer.on_odometry(moving(1.0, 0.0, 0.0))
    assert turn is not None
    assert turn.behavior is Behavior.TURN_TO
    assert turn.theta == pytest.approx(math.pi / 2)
    assert sequencer.on_odometry(moving(1.0, 0.0, math.pi / 2)) is None  # still turning

    go = sequencer.on_odometry(at_rest(1.0, 0.0, math.pi / 2 - 0.05))
    assert go is not None
    assert go.behavior is Behavior.GO_TO
    assert (go.x, go.y) == pytest.approx((1.0, 0.5))

    assert sequencer.on_odometry(at_rest(1.0, 0.49, math.pi / 2)) == STOP
    assert sequencer.status is MissionStatus.SUCCEEDED
    assert sequencer.command == STOP


def test_the_command_in_force_is_the_one_of_the_running_step(sequencer):
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))

    assert sequencer.command == Command(Behavior.TWIST, linear=0.2)


def test_a_code_activated_in_the_first_step_aborts(sequencer):
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    assert sequencer.on_state(NO_CODES) is None

    assert sequencer.on_state(FAULT) == STOP
    assert sequencer.status is MissionStatus.ABORTED
    assert sequencer.command == STOP


def test_a_reflex_activated_also_aborts(sequencer):
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))

    assert sequencer.on_state(frozenset({Code.REFLEX_FRONT_OBSTACLE})) == STOP


def test_a_code_active_before_the_mission_counts_as_activated_when_it_begins(sequencer):
    sequencer.on_state(FAULT)
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))

    assert sequencer.on_state(FAULT) == STOP


def test_after_the_end_nothing_restarts_the_mission(sequencer):
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    sequencer.on_state(FAULT)

    assert sequencer.on_odometry(moving(1.0, 0.0, 0.0)) is None
    assert sequencer.on_state(NO_CODES) is None
    assert sequencer.on_state(FAULT) is None
    assert sequencer.status is MissionStatus.ABORTED
