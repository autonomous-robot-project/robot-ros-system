from robot_sequencer.mission import DistanceFromStart, GoTo, HeadingReached, TurnTo, Twist
from robot_sequencer.missions import D_1, MISSIONS, V_1, build_v01_mission


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
