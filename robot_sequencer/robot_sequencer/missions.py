"""
Missions of the verifications, as data (ADR 032), chosen by the parameter `mission` of the node.

Criticality: T2, product (robot-docs codigo.md).

Each mission is built by a function, so every run gets new targets: an Offset is resolved once.

V-01, sequencing rule (failure is the activation of a code in the State; abort is the default):

| State | Behavior | Success | Failure |
| S1 | TWIST (v₁, 0) | distance ≥ d₁ (odometry) -> S2 | S4 |
| S2 | TURN_TO (θ₁) | |θ₁ − θ| ≤ ε_θ and ω = 0 (odometry) -> S3 | S4 |
| S3 | GO_TO (x₁, y₁) | distance to the target ≤ ε_d and v = 0 (odometry) -> S4 | S4 |
| S4 | STOP = TWIST (0, 0) (ADR 030) | end | - |

θ₁ = heading at the end of S1 + π/2; (x₁, y₁) = position at the end of S1 + d₂ (cos θ₁, sin θ₁).
S4 is the TWIST(0, 0) the sequencer sends at the end or the abort of a mission.

V-07, repetition rule: TWIST(v_r, 0) -> activation of the test code -> TWIST(0, 0) -> its
deactivation -> TWIST(v_r, 0). The controller activates FAULT_TEST every T_ev and clears it when
the TWIST(0, 0) arrives (ADR 049); any other code aborts, the default.
"""

from collections.abc import Callable
import math

from robot_sequencer.mission import (
    Code,
    DistanceFromStart,
    GoTo,
    HeadingTo,
    Mission,
    Never,
    Offset,
    Recover,
    Step,
    TurnTo,
    Twist,
    wait_until_clear,
)

# Parameters of V-01.
V_1: float = 0.2  # m/s, v₁: d₁ / v₁ = 5 s, time to lock a wheel by hand during S1
D_1: float = 1.0  # m, d₁
D_2: float = 0.5  # m, d₂
TURN_1: float = math.pi / 2  # rad, θ₁ minus the heading at the end of S1


def build_v01_mission() -> Mission:
    corner = Offset(turn=TURN_1, distance=D_2)
    return Mission(
        steps=[
            Step(Twist(V_1, 0.0), until=DistanceFromStart(D_1)),  # S1
            Step(TurnTo(HeadingTo(corner))),  # S2
            Step(GoTo(corner)),  # S3
        ]
    )


# Parameters of V-07.
V_R: float = 0.2  # m/s, v_r, commanded before each event


def build_v07_mission() -> Mission:
    return Mission(
        steps=[Step(Twist(V_R, 0.0), until=Never())],
        reactions={Code.FAULT_TEST: Recover([wait_until_clear(Code.FAULT_TEST)])},
    )


MISSIONS: dict[str, Callable[[], Mission]] = {
    'v01': build_v01_mission,
    'v07': build_v07_mission,
}
