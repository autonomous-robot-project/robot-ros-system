from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import (
    STOP,
    Behavior,
    Code,
    Command,
    DistanceFromStart,
    GoTo,
    HeadingReached,
    OdometryReading,
    TurnTo,
    Twist,
)
from robot_sequencer.missions import D_1, MISSIONS, V_1, V_R, build_v01_mission, build_v07_mission
from robot_sequencer.sequencer import MissionStatus, Sequencer


def test_v01_has_the_steps_of_its_sequencing_rule():
    s1, s2, s3 = build_v01_mission().steps

    assert s1.behavior == Twist(V_1, 0.0)
    assert s1.termination == DistanceFromStart(D_1)
    assert isinstance(s2.behavior, TurnTo)
    assert s2.termination == HeadingReached()
    assert isinstance(s3.behavior, GoTo)


def test_v01_turns_toward_the_corner_it_then_goes_to():
    _, s2, s3 = build_v01_mission().steps

    assert isinstance(s2.behavior, TurnTo)
    assert isinstance(s3.behavior, GoTo)
    assert s2.behavior.heading.target is s3.behavior.target


def test_each_build_gets_a_new_corner():
    _, _, first = build_v01_mission().steps
    _, _, second = build_v01_mission().steps

    assert isinstance(first.behavior, GoTo)
    assert isinstance(second.behavior, GoTo)
    assert first.behavior.target is not second.behavior.target


def test_the_node_parameter_names_v01():
    assert MISSIONS['v01'] is build_v01_mission


def test_v07_repeats_twist_vr_then_stop_until_the_test_code_clears():
    sequencer = Sequencer(build_v07_mission())
    forward = sequencer.on_odometry(OdometryReading(Pose(Point(0.0, 0.0), 0.0), 0.0, 0.0))
    test_code = frozenset({Code.FAULT_TEST})

    assert forward == Command(Behavior.TWIST, linear=V_R)
    for _ in range(2):
        assert sequencer.on_state(test_code) == STOP
        assert sequencer.on_state(frozenset()) == forward


def test_v07_aborts_on_any_other_code():
    sequencer = Sequencer(build_v07_mission())
    sequencer.on_odometry(OdometryReading(Pose(Point(0.0, 0.0), 0.0), 0.0, 0.0))

    assert sequencer.on_state(frozenset({Code.FAULT_DEADLINE_MISSED})) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_the_node_parameter_names_v07():
    assert MISSIONS['v07'] is build_v07_mission
