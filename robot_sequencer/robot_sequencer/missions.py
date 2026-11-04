"""
Missions of the verifications, as data (ADR 032), chosen by the parameter `mission` of the node.

Criticality: T2, product (robot-docs codigo.md).

Each mission is built by a function, so every run gets new targets: an Offset is resolved once.

V-01, sequencing rule (failure is the activation of a code of the State or an event of the
sequencer, except HEARTBEAT_LOST; abort is the default):

| State | Behavior | Success | Failure |
| S1 | TWIST (v₁, 0) | distance ≥ d₁ (odometry) -> S2 | S4 |
| S2 | TURN_TO (θ₁) | |θ₁ − θ| ≤ ε_θ at rest (v = 0 and ω = 0, odometry) -> S3 | S4 |
| S3 | GO_TO (x₁, y₁) | distance to target ≤ ε_d at rest (v = 0 and ω = 0, odometry) -> S4 | S4 |
| S4 | STOP = TWIST (0, 0) (ADR 030) | end | - |

θ₁ = heading at the end of S1 + Δθ₁; (x₁, y₁) = position at the end of S1 + d₂ (cos θ₁, sin θ₁).
S4 is the TWIST(0, 0) the sequencer sends at the end or the abort of a mission.

The missions of V-01 and V-07 run without the notebook, so they ignore HEARTBEAT_LOST (ADR 038);
any other event of the sequencer aborts them.

V-08: the mission of V-07 under load. Under C2 the link may lag, and the expected effect is the
expiry of the command (RNF-02), which also leaves the echo in TWIST(0, 0): the mission ignores
LINK_LOST and COMMAND_NOT_ECHOED as well (ADR 038).

V-07, repetition rule: TWIST(v_r, 0) -> activation of the test code -> TWIST(0, 0) -> its
deactivation -> TWIST(v_r, 0). The controller activates FAULT_TEST every T_ev and clears it when
the TWIST(0, 0) arrives (ADR 049); any other code aborts, the default.

Mission by plan (V-04): one pair TURN_TO(heading to the vertex) -> GO_TO(vertex) per segment of
the plan (ADRs 015, 024 and 034), the vertices in map (ADR 035). Table of ADR 038, with abort
where its ladder is not implemented (a debt of V-09):

| Condition | TURN_TO | GO_TO |
| Front obstacle, front sensor fault | ignore | abort (ADR 038: ladder of wait_until_clear) |
| Deadline missed | ignore | ignore |
| Any other code | abort | abort |
"""

from collections.abc import Callable, Sequence
import math

from robot_sequencer.geometry import Point
from robot_sequencer.mission import (
    Condition,
    DistanceFromStart,
    GoTo,
    HeadingTo,
    Ignore,
    MapPoint,
    Mission,
    Never,
    Offset,
    Recover,
    Step,
    TurnTo,
    Twist,
    wait_until_clear,
)

# The missions of the verifications run without the heartbeat of the notebook (ADR 038).
WITHOUT_NOTEBOOK = {Condition.HEARTBEAT_LOST: Ignore()}

# Parameters of V-01.
V_1: float = 0.2  # m/s, v₁: d₁ / v₁ = 5 s, time to lock a wheel by hand during S1
D_1: float = 1.0  # m, d₁
D_2: float = 0.5  # m, d₂
DELTA_THETA_1: float = math.pi / 2  # rad, Δθ₁: θ₁ minus the heading at the end of S1


def build_v01_mission() -> Mission:
    corner = Offset(turn=DELTA_THETA_1, distance=D_2)
    return Mission(
        steps=[
            Step(Twist(V_1, 0.0), until=DistanceFromStart(D_1)),  # S1
            Step(TurnTo(HeadingTo(corner))),  # S2
            Step(GoTo(corner)),  # S3
        ],
        reactions=WITHOUT_NOTEBOOK,
    )


# Parameters of V-07.
V_R: float = 0.2  # m/s, v_r, commanded before each event


def build_v07_mission() -> Mission:
    return Mission(
        steps=[Step(Twist(V_R, 0.0), until=Never())],
        reactions={
            **WITHOUT_NOTEBOOK,
            Condition.FAULT_TEST: Recover([wait_until_clear(Condition.FAULT_TEST)]),
        },
    )


# The front reflex and the fault of its sensor allow a turn in place (ADR 021).
TURN_IN_PLACE_REACTIONS = {
    Condition.REFLEX_FRONT_OBSTACLE: Ignore(),
    Condition.FAULT_FRONT_SENSOR: Ignore(),
}


def build_plan_mission(vertices: Sequence[Point]) -> Mission:
    """Return the mission of a plan, from its vertices in map (plan.py)."""
    steps: list[Step] = []
    for vertex in vertices:
        target = MapPoint(vertex)
        steps.append(Step(TurnTo(HeadingTo(target)), reactions=TURN_IN_PLACE_REACTIONS))
        steps.append(Step(GoTo(target)))
    return Mission(steps, reactions={Condition.FAULT_DEADLINE_MISSED: Ignore()})


def build_v08_mission() -> Mission:
    v07 = build_v07_mission()
    under_load = {Condition.LINK_LOST: Ignore(), Condition.COMMAND_NOT_ECHOED: Ignore()}
    return Mission(v07.steps, reactions={**v07.reactions, **under_load})


MISSIONS: dict[str, Callable[[], Mission]] = {
    'v01': build_v01_mission,
    'v07': build_v07_mission,
    'v08': build_v08_mission,
}
