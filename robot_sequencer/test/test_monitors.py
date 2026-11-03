"""The events of the sequencer (ADR 037): echo of the command, age of the State, heartbeat."""

import pytest
from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import (
    STOP,
    Behavior,
    Command,
    Condition,
    Mission,
    Never,
    OdometryReading,
    StateReading,
    Step,
    Twist,
    round_value_to_float32,
)
from robot_sequencer.missions import build_v01_mission, build_v07_mission
from robot_sequencer.sequencer import (
    ECHO_TIMEOUT_S,
    HEARTBEAT_TIMEOUT_S,
    LINK_TIMEOUT_S,
    MissionStatus,
    Sequencer,
)

NO_CONDITIONS: frozenset[Condition] = frozenset()
MARGIN_S = 0.01
FORWARD = Command(Behavior.TWIST, linear=0.2)
FORWARD_ECHO = FORWARD.round_to_float32()


def started(mission: Mission, heartbeat: bool = True) -> Sequencer:
    """Return the sequencer in its first step, with a State and a heartbeat at time 0."""
    sequencer = Sequencer(mission)
    if heartbeat:
        sequencer.on_heartbeat(0.0)
    sequencer.on_odometry(OdometryReading(Pose(Point(0.0, 0.0), 0.0), linear=0.0, angular=0.0))
    sequencer.on_state(StateReading(NO_CONDITIONS, echo=sequencer.command.round_to_float32()), 0.0)
    return sequencer


def forward_mission() -> Mission:
    return Mission([Step(Twist(0.2, 0.0), until=Never())])


def test_echo_rounded_to_float32_is_the_command():
    command = Command(Behavior.TURN_TO, theta=0.1)  # 0.1 has no exact float32
    echo = Command(Behavior.TURN_TO, theta=round_value_to_float32(0.1))
    assert command.is_echoed_by(echo)
    assert not command.is_echoed_by(Command(Behavior.TURN_TO, theta=0.2))


def test_echo_that_differs_up_to_t_eco_is_tolerated():
    sequencer = started(forward_mission())
    assert sequencer.on_state(StateReading(NO_CONDITIONS, echo=STOP), 0.01) is None
    assert sequencer.on_state(StateReading(NO_CONDITIONS, echo=STOP), ECHO_TIMEOUT_S) is None
    assert (
        sequencer.on_state(StateReading(NO_CONDITIONS, echo=FORWARD_ECHO), ECHO_TIMEOUT_S + 0.01)
        is None
    )
    assert sequencer.status is MissionStatus.RUNNING


def test_echo_that_differs_beyond_t_eco_aborts():
    sequencer = started(forward_mission())
    sequencer.on_state(StateReading(NO_CONDITIONS, echo=STOP), 0.01)
    late = 0.01 + ECHO_TIMEOUT_S + MARGIN_S

    assert sequencer.on_state(StateReading(NO_CONDITIONS, echo=STOP), late) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_state_younger_than_t_link_keeps_the_mission():
    sequencer = started(forward_mission())
    assert sequencer.on_timer(LINK_TIMEOUT_S - MARGIN_S) is None
    assert sequencer.status is MissionStatus.RUNNING


def test_state_older_than_t_link_aborts_on_the_timer():
    sequencer = started(forward_mission())
    assert sequencer.on_timer(LINK_TIMEOUT_S + MARGIN_S) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_age_of_the_state_counts_from_the_first_state():
    sequencer = Sequencer(forward_mission())
    sequencer.on_heartbeat(0.0)
    sequencer.on_odometry(OdometryReading(Pose(Point(0.0, 0.0), 0.0), linear=0.0, angular=0.0))
    assert sequencer.on_timer(LINK_TIMEOUT_S + MARGIN_S) is None


def test_heartbeat_older_than_t_hb_aborts():
    sequencer = started(forward_mission())
    sequencer.on_state(StateReading(NO_CONDITIONS, echo=FORWARD_ECHO), HEARTBEAT_TIMEOUT_S)

    late = HEARTBEAT_TIMEOUT_S + MARGIN_S
    assert sequencer.on_state(StateReading(NO_CONDITIONS, echo=FORWARD_ECHO), late) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_mission_does_not_begin_without_heartbeat():
    sequencer = started(forward_mission(), heartbeat=False)
    assert sequencer.status is MissionStatus.ABORTED
    assert sequencer.command == STOP


@pytest.mark.parametrize('build_mission', [build_v01_mission, build_v07_mission])
def test_missions_of_the_verifications_run_without_heartbeat(build_mission):
    sequencer = started(build_mission(), heartbeat=False)
    echo = sequencer.command.round_to_float32()
    sequencer.on_state(
        StateReading(NO_CONDITIONS, echo=echo), HEARTBEAT_TIMEOUT_S
    )  # the link is up

    assert sequencer.on_timer(HEARTBEAT_TIMEOUT_S + MARGIN_S) is None
    assert sequencer.status is MissionStatus.RUNNING


@pytest.mark.parametrize('build_mission', [build_v01_mission, build_v07_mission])
def test_missions_of_the_verifications_abort_on_link_lost(build_mission):
    sequencer = started(build_mission(), heartbeat=False)
    assert sequencer.on_timer(LINK_TIMEOUT_S + MARGIN_S) == STOP
    assert sequencer.status is MissionStatus.ABORTED
