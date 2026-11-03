"""The mission by plan on the engine: a pair TURN_TO -> GO_TO per vertex in map (V-04)."""

import math

import pytest
from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import (
    STOP,
    Abort,
    Behavior,
    Condition,
    GoTo,
    Ignore,
    MapPoint,
    Mission,
    OdometryReading,
    Step,
    TurnTo,
)
from robot_sequencer.missions import build_plan_mission
from robot_sequencer.sequencer import MissionStatus, Sequencer

IDENTITY = Pose(Point(0.0, 0.0), 0.0)  # map -> odom static and identity on the bench (V-04)
VERTICES = [Point(1.0, 0.0), Point(1.0, 1.0)]


def at_rest(x: float, y: float, theta: float) -> OdometryReading:
    return OdometryReading(Pose(Point(x, y), theta), linear=0.0, angular=0.0)


def test_each_vertex_gives_a_turn_to_and_a_go_to_aimed_at_it():
    steps = build_plan_mission(VERTICES).steps

    assert [type(step.behavior) for step in steps] == [TurnTo, GoTo, TurnTo, GoTo]
    for turn, go, vertex in zip(steps[::2], steps[1::2], VERTICES, strict=True):
        assert turn.behavior.heading.target == MapPoint(vertex)
        assert go.behavior.target == MapPoint(vertex)


@pytest.mark.parametrize(
    'condition', [Condition.REFLEX_FRONT_OBSTACLE, Condition.FAULT_FRONT_SENSOR]
)
def test_front_conditions_are_ignored_in_the_turn_and_abort_the_go_to(condition):
    mission = build_plan_mission(VERTICES)
    turn, go = mission.steps[:2]

    assert mission.get_reaction(turn, condition) == Ignore()
    assert mission.get_reaction(go, condition) == Abort()


def test_deadline_missed_is_ignored_and_other_conditions_abort():
    mission = build_plan_mission(VERTICES)
    for step in mission.steps[:2]:
        assert mission.get_reaction(step, Condition.FAULT_DEADLINE_MISSED) == Ignore()
        assert mission.get_reaction(step, Condition.FAULT_COGNIZANT) == Abort()


def test_steps_aim_at_the_vertices_converted_to_odom():
    sequencer = Sequencer(build_plan_mission(VERTICES), get_map_to_odom=lambda: IDENTITY)

    turn = sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    assert turn is not None and turn.behavior is Behavior.TURN_TO
    assert turn.theta == pytest.approx(0.0)
    go = sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    assert go is not None and (go.behavior, go.x, go.y) == (Behavior.GO_TO, 1.0, 0.0)


def test_vertex_is_converted_again_with_the_latest_map_to_odom_at_each_entry():
    map_to_odom = [IDENTITY]
    sequencer = Sequencer(build_plan_mission(VERTICES), get_map_to_odom=lambda: map_to_odom[0])
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # TURN_TO with the identity

    map_to_odom[0] = Pose(Point(0.1, 0.0), 0.0)  # the SLAM corrected the drift by 0.1 m
    go = sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))

    assert go is not None and (go.x, go.y) == pytest.approx((0.9, 0.0))


def test_without_map_to_odom_the_mission_aborts_with_twist_zero():
    sequencer = Sequencer(build_plan_mission(VERTICES), get_map_to_odom=lambda: None)

    assert sequencer.on_odometry(at_rest(0.0, 0.0, 0.0)) == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_tf_missing_accepts_only_abort_in_the_mission_and_in_the_step():
    with pytest.raises(ValueError):
        Mission([], reactions={Condition.TF_MISSING: Ignore()})
    with pytest.raises(ValueError):
        Step(GoTo(MapPoint(Point(1.0, 0.0))), reactions={Condition.TF_MISSING: Ignore()})


def test_last_vertex_reached_ends_the_mission_with_twist_zero():
    sequencer = Sequencer(build_plan_mission([Point(1.0, 0.0)]), get_map_to_odom=lambda: IDENTITY)
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # TURN_TO, already aligned
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # GO_TO

    assert sequencer.on_odometry(at_rest(1.0, 0.0, math.pi / 4)) == STOP
    assert sequencer.status is MissionStatus.SUCCEEDED


# --- Life of the mission by plan: idle, planning, start, failure, preemption ---


def test_without_a_mission_it_is_idle_with_twist_zero():
    sequencer = Sequencer(None, get_map_to_odom=lambda: IDENTITY)

    assert sequencer.on_odometry(at_rest(0.0, 0.0, 0.0)) is None
    assert sequencer.status is MissionStatus.IDLE
    assert sequencer.command == STOP


def test_a_goal_holds_twist_zero_until_its_plan_begins_from_the_latest_pose():
    sequencer = Sequencer(None, get_map_to_odom=lambda: IDENTITY)
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))

    assert sequencer.wait_for_plan() == STOP
    assert sequencer.status is MissionStatus.PLANNING
    turn = sequencer.start(build_plan_mission(VERTICES))
    assert turn is not None and turn.behavior is Behavior.TURN_TO
    assert sequencer.status is MissionStatus.RUNNING


def test_a_plan_before_the_first_odometry_begins_on_it():
    sequencer = Sequencer(None, get_map_to_odom=lambda: IDENTITY)
    sequencer.wait_for_plan()

    assert sequencer.start(build_plan_mission(VERTICES)) is None
    assert sequencer.status is MissionStatus.WAITING
    turn = sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    assert turn is not None and turn.behavior is Behavior.TURN_TO


def test_a_failed_plan_aborts_with_twist_zero():
    sequencer = Sequencer(None, get_map_to_odom=lambda: IDENTITY)
    sequencer.wait_for_plan()

    assert sequencer.on_plan_failed() == STOP
    assert sequencer.status is MissionStatus.ABORTED


def test_a_new_goal_preempts_the_mission_and_the_new_plan_replaces_it():
    sequencer = Sequencer(None, get_map_to_odom=lambda: IDENTITY)
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))
    sequencer.start(build_plan_mission(VERTICES))

    assert sequencer.wait_for_plan() == STOP
    assert sequencer.command == STOP
    turn = sequencer.start(build_plan_mission([Point(0.0, 1.0)]))
    assert turn is not None and turn.theta == pytest.approx(math.pi / 2)


def test_plan_failed_accepts_only_abort():
    with pytest.raises(ValueError):
        Mission([], reactions={Condition.PLAN_FAILED: Ignore()})
