"""The mission by plan on the engine: a pair TURN_TO -> GO_TO per vertex in map (V-04)."""

import math

import pytest
from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import (
    STOP,
    Abort,
    Behavior,
    Code,
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


@pytest.mark.parametrize('code', [Code.REFLEX_FRONT_OBSTACLE, Code.FAULT_FRONT_SENSOR])
def test_front_codes_are_ignored_in_the_turn_and_abort_the_go_to(code):
    mission = build_plan_mission(VERTICES)
    turn, go = mission.steps[:2]

    assert mission.get_reaction(turn, code) == Ignore()
    assert mission.get_reaction(go, code) == Abort()


def test_deadline_missed_is_ignored_and_other_codes_abort():
    mission = build_plan_mission(VERTICES)
    for step in mission.steps[:2]:
        assert mission.get_reaction(step, Code.FAULT_DEADLINE_MISSED) == Ignore()
        assert mission.get_reaction(step, Code.FAULT_COGNIZANT) == Abort()


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
        Mission([], reactions={Code.TF_MISSING: Ignore()})
    with pytest.raises(ValueError):
        Step(GoTo(MapPoint(Point(1.0, 0.0))), reactions={Code.TF_MISSING: Ignore()})


def test_last_vertex_reached_ends_the_mission_with_twist_zero():
    sequencer = Sequencer(build_plan_mission([Point(1.0, 0.0)]), get_map_to_odom=lambda: IDENTITY)
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # TURN_TO, already aligned
    sequencer.on_odometry(at_rest(0.0, 0.0, 0.0))  # GO_TO

    assert sequencer.on_odometry(at_rest(1.0, 0.0, math.pi / 4)) == STOP
    assert sequencer.status is MissionStatus.SUCCEEDED
