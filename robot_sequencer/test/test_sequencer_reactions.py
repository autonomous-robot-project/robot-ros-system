"""The engine reacting to codes: recover, ignore and abort, on a mission shaped like V-07."""

from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import (
    STOP,
    Behavior,
    Code,
    Command,
    Ignore,
    Mission,
    Never,
    OdometryReading,
    Recover,
    StateReading,
    Step,
    Twist,
    wait_until_clear,
)
from robot_sequencer.sequencer import MissionStatus, Sequencer


def report(sequencer: Sequencer, codes: frozenset[Code], now: float = 0.0) -> Command | None:
    """Send a State at now that echoes the command in force."""
    return sequencer.on_state(StateReading(codes, echo=sequencer.command.round_to_float32()), now)


TEST = frozenset({Code.FAULT_TEST})
NO_CODES: frozenset[Code] = frozenset()
FORWARD = Command(Behavior.TWIST, linear=0.2)


def repetition_mission(**step_reactions) -> Mission:
    """TWIST(0.2, 0) with no termination; FAULT_TEST -> TWIST(0, 0) until it clears."""
    step = Step(Twist(0.2, 0.0), until=Never(), reactions=step_reactions.get('reactions', {}))
    return Mission(
        steps=[step],
        reactions={Code.FAULT_TEST: Recover([wait_until_clear(Code.FAULT_TEST)])},
    )


def started(mission: Mission) -> Sequencer:
    sequencer = Sequencer(mission)
    sequencer.on_heartbeat(0.0)
    sequencer.on_odometry(OdometryReading(Pose(Point(0.0, 0.0), 0.0), linear=0.0, angular=0.0))
    report(sequencer, NO_CODES)
    return sequencer


def test_an_activation_runs_the_recovery_and_the_clearing_returns_to_the_step():
    sequencer = started(repetition_mission())

    assert report(sequencer, TEST) == STOP
    assert report(sequencer, TEST) is None  # the wait step ignores its own code
    assert report(sequencer, NO_CODES) == FORWARD
    assert sequencer.status is MissionStatus.RUNNING


def test_the_repetition_works_event_after_event():
    sequencer = started(repetition_mission())

    for _ in range(3):
        assert report(sequencer, TEST) == STOP
        assert report(sequencer, NO_CODES) == FORWARD


def test_an_odometry_does_not_end_a_wait_step():
    sequencer = started(repetition_mission())
    report(sequencer, TEST)

    reading = OdometryReading(Pose(Point(0.0, 0.0), 0.0), linear=0.0, angular=0.0)
    assert sequencer.on_odometry(reading) is None
    assert sequencer.command == STOP


def test_another_code_during_the_wait_aborts_by_the_default():
    sequencer = started(repetition_mission())
    report(sequencer, TEST)

    assert report(sequencer, TEST | {Code.FAULT_COGNIZANT}) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_abort_comes_before_recover_when_both_codes_turn_on_together():
    sequencer = started(repetition_mission())

    assert report(sequencer, TEST | {Code.FAULT_COGNIZANT}) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_a_step_that_ignores_the_code_keeps_running():
    sequencer = started(repetition_mission(reactions={Code.FAULT_TEST: Ignore()}))

    assert report(sequencer, TEST) is None
    assert sequencer.command == FORWARD


def test_a_recovery_with_two_steps_runs_both_before_returning():
    a, b = Code.FAULT_TEST, Code.FAULT_FRONT_SENSOR
    step = Step(Twist(0.2, 0.0), until=Never())
    mission = Mission(
        steps=[step],
        reactions={a: Recover([wait_until_clear(a), wait_until_clear(b)]), b: Ignore()},
    )
    sequencer = started(mission)

    assert report(sequencer, frozenset({a, b})) == STOP  # a recovers, b is ignored
    assert report(sequencer, frozenset({b})) == STOP  # a cleared: the wait for b begins
    assert report(sequencer, NO_CODES) == FORWARD  # b cleared: back to the step
