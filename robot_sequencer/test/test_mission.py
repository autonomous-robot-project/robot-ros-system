import math

import pytest
from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import (
    EPSILON_D,
    EPSILON_THETA,
    Abort,
    Behavior,
    Cleared,
    Command,
    Condition,
    DistanceFromStart,
    GoTo,
    HeadingReached,
    HeadingTo,
    Ignore,
    Mission,
    Never,
    OdometryReading,
    Offset,
    PositionReached,
    Recover,
    Step,
    TurnTo,
    Twist,
    wait_until_clear,
)

ORIGIN = Pose(Point(0.0, 0.0), 0.0)


def at_rest(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.0, angular=0.0)


def moving(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.1, angular=0.0)


def test_offset_is_resolved_once_from_the_first_pose():
    corner = Offset(turn=math.pi / 2, distance=0.5)
    end_of_twist = Pose(Point(1.0, 0.0), 0.0)

    first = corner.resolve_point(end_of_twist)
    later = corner.resolve_point(Pose(Point(1.0, 0.1), math.pi / 2))

    assert first.x == pytest.approx(1.0)
    assert first.y == pytest.approx(0.5)
    assert later == first


def test_heading_to_the_offset_at_its_pose_is_the_heading_plus_the_turn():
    # V-01: θ₁ = heading at the end of S1 + Δθ₁.
    end_of_twist = Pose(Point(1.0, 0.0), 0.3)
    corner = Offset(turn=math.pi / 2, distance=0.5)

    assert HeadingTo(corner).resolve_heading(end_of_twist) == pytest.approx(0.3 + math.pi / 2)


def test_turn_to_sends_the_heading_wrapped():
    pose = Pose(Point(0.0, 0.0), 3.0)
    corner = Offset(turn=math.pi / 2, distance=0.5)

    command = TurnTo(HeadingTo(corner)).resolve(pose)

    assert command.behavior is Behavior.TURN_TO
    assert command.theta == pytest.approx(3.0 + math.pi / 2 - 2 * math.pi)


def test_go_to_sends_the_point_of_the_offset():
    corner = Offset(turn=math.pi / 2, distance=0.5)

    command = GoTo(corner).resolve(ORIGIN)

    assert command.behavior is Behavior.GO_TO
    assert (command.x, command.y) == pytest.approx((0.0, 0.5))


def test_twist_sends_its_speeds():
    assert Twist(0.2, 0.0).resolve(ORIGIN) == Command(Behavior.TWIST, linear=0.2)


@pytest.mark.parametrize(('x', 'met'), [(0.99, False), (1.0, True), (1.2, True)])
def test_distance_from_start_is_met_at_the_distance(x, met):
    reading = moving(x, 0.0, 0.0)

    assert DistanceFromStart(1.0).is_met(ORIGIN, Twist(0.2, 0.0).resolve(ORIGIN), reading) is met


def test_distance_from_start_is_the_displacement_not_the_path():
    start = Pose(Point(1.0, 1.0), 0.0)
    back_near_start = moving(1.5, 1.0, math.pi)

    assert not DistanceFromStart(1.0).is_met(
        start, Twist(0.2, 0.0).resolve(start), back_near_start
    )


@pytest.mark.parametrize(
    ('reading', 'met'),
    [
        (at_rest(0.0, 0.0, math.pi / 2 - 0.9 * EPSILON_THETA), True),
        (at_rest(0.0, 0.0, math.pi / 2 - 1.1 * EPSILON_THETA), False),
        (moving(0.0, 0.0, math.pi / 2), False),
    ],
    ids=['within at rest', 'outside at rest', 'within moving'],
)
def test_heading_reached_needs_the_tolerance_and_rest(reading, met):
    command = Command(Behavior.TURN_TO, theta=math.pi / 2)

    assert HeadingReached().is_met(ORIGIN, command, reading) is met


def test_heading_reached_measures_the_error_across_plus_minus_pi():
    command = Command(Behavior.TURN_TO, theta=math.pi - 0.05)

    assert HeadingReached().is_met(ORIGIN, command, at_rest(0.0, 0.0, -math.pi + 0.05))


@pytest.mark.parametrize(
    ('reading', 'met'),
    [
        (at_rest(1.0, 0.5 - 0.9 * EPSILON_D, 0.0), True),
        (at_rest(1.0, 0.5 - 1.1 * EPSILON_D, 0.0), False),
        (moving(1.0, 0.5, 0.0), False),
    ],
    ids=['within at rest', 'outside at rest', 'within moving'],
)
def test_position_reached_needs_the_tolerance_and_rest(reading, met):
    command = Command(Behavior.GO_TO, x=1.0, y=0.5)

    assert PositionReached().is_met(ORIGIN, command, reading) is met


def test_a_reading_is_at_rest_only_with_both_speeds_exactly_zero():
    assert at_rest(0.0, 0.0, 0.0).is_at_rest
    assert not OdometryReading(ORIGIN, linear=0.0, angular=1e-6).is_at_rest


def test_a_twist_step_needs_an_explicit_termination():
    with pytest.raises(ValueError):
        Step(Twist(0.2, 0.0))


def test_turn_to_and_go_to_steps_take_the_default_termination():
    corner = Offset(turn=math.pi / 2, distance=0.5)

    assert Step(TurnTo(HeadingTo(corner))).termination == HeadingReached()
    assert Step(GoTo(corner)).termination == PositionReached()


def test_an_explicit_termination_replaces_the_default():
    corner = Offset(turn=math.pi / 2, distance=0.5)

    assert Step(GoTo(corner), until=DistanceFromStart(0.1)).termination == DistanceFromStart(0.1)


def test_never_is_never_met():
    assert not Never().is_met(ORIGIN, Twist(0.2, 0.0).resolve(ORIGIN), moving(9.0, 9.0, 0.0))


def test_wait_until_clear_holds_twist_zero_and_ignores_its_own_code():
    step = wait_until_clear(Condition.FAULT_TEST)

    assert step.behavior == Twist(0.0, 0.0)
    assert step.termination == Cleared(Condition.FAULT_TEST)
    assert step.reactions == {Condition.FAULT_TEST: Ignore()}


def test_the_reaction_of_the_step_comes_before_the_one_of_the_mission():
    step = Step(Twist(0.2, 0.0), until=Never(), reactions={Condition.FAULT_TEST: Ignore()})
    recover = Recover([wait_until_clear(Condition.FAULT_TEST)])
    mission = Mission(steps=[step], reactions={Condition.FAULT_TEST: recover})

    assert mission.get_reaction(step, Condition.FAULT_TEST) == Ignore()


def test_the_reaction_of_the_mission_applies_when_the_step_has_none():
    step = Step(Twist(0.2, 0.0), until=Never())
    recover = Recover([wait_until_clear(Condition.FAULT_TEST)])
    mission = Mission(steps=[step], reactions={Condition.FAULT_TEST: recover})

    assert mission.get_reaction(step, Condition.FAULT_TEST) == recover


def test_a_code_in_no_table_aborts():
    step = Step(Twist(0.2, 0.0), until=Never())

    assert Mission(steps=[step]).get_reaction(step, Condition.FAULT_COGNIZANT) == Abort()
