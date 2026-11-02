"""
Vocabulary of the missions: steps of primitive behaviors, their targets and terminations.

Criticality: T2, product (robot-docs codigo.md).

A mission is data (ADR 032): a list of steps, each with a behavior of ADR 024, parameters computed
at the entry of the step from the current pose (and the latest map -> odom for a vertex of the
plan, ADR 035), and a termination tested on the message that
changes its input: the pose on each Odometry, a code on each State (ADR 033). The reaction to the
activation of a code comes from the table of the step, else of the mission, else abort.
Domain without ROS (ADR 047): the node converts the messages to these types.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, auto
import math
import struct

from robot_sequencer.geometry import Point, Pose, wrap_angle

# Tolerances of the default terminations (V-01): above the worst final error of the controller
# plus the resolution of the odometry, measured on the bench (robot-docs nota 0014):
# 3.6 deg + 3.9 deg per edge of one wheel, and 6 mm + 10 mm per pulse.
EPSILON_THETA: float = math.radians(10.0)  # rad, ε_θ
EPSILON_D: float = 0.05  # m, ε_d
# Largest turn of one TURN_TO, so no target reaches 180 deg from the heading (ADRs 024 and 036):
# the starting point of ADR 036, with margin over the 4.5 deg per pulse of the odometry.
THETA_MAX: float = math.radians(170.0)  # rad, θ_max


class Behavior(Enum):
    """Primitive behavior of the controller, with the values of Command.msg (ADR 024)."""

    TWIST = 0
    TURN_TO = 1
    GO_TO = 2


def round_value_to_float32(value: float) -> float:
    return struct.unpack('f', struct.pack('f', value))[0]


@dataclass(frozen=True)
class Command:
    """Behavior and parameters sent to the controller; the unused fields are zero."""

    behavior: Behavior
    linear: float = 0.0  # m/s, TWIST
    angular: float = 0.0  # rad/s, TWIST
    theta: float = 0.0  # rad in odom, TURN_TO
    x: float = 0.0  # m in odom, GO_TO
    y: float = 0.0  # m in odom, GO_TO

    def is_echoed_by(self, echo: Command) -> bool:
        return self.round_to_float32() == echo

    def round_to_float32(self) -> Command:
        """Return the command as it travels in the float32 fields of Command.msg."""
        return Command(
            self.behavior,
            linear=round_value_to_float32(self.linear),
            angular=round_value_to_float32(self.angular),
            theta=round_value_to_float32(self.theta),
            x=round_value_to_float32(self.x),
            y=round_value_to_float32(self.y),
        )


# TWIST(0, 0): the command when no mission runs, after its end and after an abort (ADR 030).
STOP = Command(Behavior.TWIST)


def limit_turn(command: Command, pose: Pose) -> Command:
    """Return the command with a TURN_TO above θ_max cut to θ_max, toward the target (ADR 036)."""
    turn = wrap_angle(command.theta - pose.theta)
    if command.behavior is not Behavior.TURN_TO or abs(turn) <= THETA_MAX:
        return command
    return replace(command, theta=wrap_angle(pose.theta + math.copysign(THETA_MAX, turn)))


@dataclass(frozen=True)
class OdometryReading:
    pose: Pose
    linear: float  # m/s
    angular: float  # rad/s

    @property
    def is_at_rest(self) -> bool:
        # The firmware publishes an exact zero once the wheel is taken as stopped (ADR 019).
        return self.linear == 0.0 and self.angular == 0.0


class Code(Enum):
    """Condition that gets a reaction: a bit of the State (ADR 025) or an event (ADR 037)."""

    FAULT_COGNIZANT = auto()
    FAULT_DEADLINE_MISSED = auto()
    FAULT_WATCHDOG_RESTART = auto()
    FAULT_FRONT_SENSOR = auto()
    FAULT_TEST = auto()
    REFLEX_FRONT_OBSTACLE = auto()
    UNKNOWN = auto()  # a bit the sequencer does not know yet: the default reaction still applies
    # Events of the sequencer.
    TF_MISSING = auto()  # no map -> odom at the entry of a step with a MapPoint (ADR 035)
    PLAN_FAILED = auto()  # planner error, empty path or no planner_server (ADR 034)
    NO_PROGRESS = auto()  # at rest out of tolerance, after moving in the step (ADR 036)
    COMMAND_NOT_ECHOED = auto()  # the echo differs from the command for more than T_eco (ADR 037)
    LINK_LOST = auto()  # no State for more than T_LINK (ADR 037)
    HEARTBEAT_LOST = auto()  # no heartbeat of the notebook for more than T_HB (ADR 037)


@dataclass(frozen=True)
class StateReading:
    codes: frozenset[Code]  # active, from the masks of the State
    echo: Command  # command in force in the controller this cycle


# Codes of a failure that leaves the step without a target: their only reaction is abort.
ABORT_ONLY_CODES: frozenset[Code] = frozenset({Code.TF_MISSING, Code.PLAN_FAILED})


@dataclass(frozen=True)
class Abort:
    """Reaction that ends the mission with TWIST(0, 0), the default for every code (ADR 032)."""


@dataclass(frozen=True)
class Ignore:
    """Reaction that leaves the step running: the code does not affect it (ADR 037)."""


@dataclass(frozen=True)
class Recover:
    """Reaction that runs its steps before the current one, then enters it again (ADR 032)."""

    steps: Sequence[Step]


Reaction = Abort | Ignore | Recover


def check_abort_only(reactions: Mapping[Code, Reaction]) -> None:
    """Raise if a table gives a code of ABORT_ONLY_CODES a reaction other than abort."""
    for code in ABORT_ONLY_CODES & reactions.keys():
        if not isinstance(reactions[code], Abort):
            raise ValueError(f'{code.name} has no target to go on with: its reaction is abort')


# Returns the latest pose of odom in map, None if the transform is not available; called only by
# a step whose target is in map, so a mission of a verification never looks it up (ADR 035).
MapToOdom = Callable[[], Pose | None]


def get_no_map_to_odom() -> Pose | None:
    return None


class MissingTransformError(Exception):
    """The map -> odom transform is not available at the entry of the step (ADR 035)."""


class Offset:
    """
    Point at a distance along the heading turned from the pose where it is first resolved.

    Resolved once and shared between the steps that use it (ADR 032): in V-01 the TURN_TO and the
    GO_TO after the TWIST aim at the same corner, computed from the pose at the end of the TWIST.
    """

    def __init__(self, turn: float, distance: float):
        self._turn = turn  # rad, added to the heading of the pose
        self._distance = distance  # m
        self._point: Point | None = None

    def resolve_point(self, pose: Pose, get_map_to_odom: MapToOdom = get_no_map_to_odom) -> Point:
        if self._point is None:
            heading = pose.theta + self._turn
            self._point = Point(
                pose.position.x + self._distance * math.cos(heading),
                pose.position.y + self._distance * math.sin(heading),
            )
        return self._point


@dataclass(frozen=True)
class MapPoint:
    """
    Vertex of the plan in map, converted to odom at the entry of each step that uses it (ADR 035).

    The conversion uses the latest map -> odom, and the target stays fixed during the step.
    """

    point: Point  # m, in map

    def resolve_point(self, pose: Pose, get_map_to_odom: MapToOdom = get_no_map_to_odom) -> Point:
        map_to_odom = get_map_to_odom()
        if map_to_odom is None:
            raise MissingTransformError
        return map_to_odom.convert_to_local(self.point)


Target = Offset | MapPoint


@dataclass(frozen=True)
class HeadingTo:
    """Heading from the current pose to a point target."""

    target: Target

    def resolve_heading(
        self, pose: Pose, get_map_to_odom: MapToOdom = get_no_map_to_odom
    ) -> float:
        return pose.position.direction_to(self.target.resolve_point(pose, get_map_to_odom))


@dataclass(frozen=True)
class DistanceFromStart:
    """The robot is at least a distance away from where the step began (V-01, S1)."""

    distance: float  # m

    def is_met(self, start: Pose, command: Command, reading: OdometryReading) -> bool:
        return start.position.distance_to(reading.pose.position) >= self.distance


@dataclass(frozen=True)
class HeadingReached:
    """Default termination of TURN_TO: heading within ε_θ, at rest (ADR 036)."""

    def is_met(self, start: Pose, command: Command, reading: OdometryReading) -> bool:
        error = wrap_angle(command.theta - reading.pose.theta)
        return reading.is_at_rest and abs(error) <= EPSILON_THETA


@dataclass(frozen=True)
class PositionReached:
    """Default termination of GO_TO: distance to the target within ε_d, at rest (ADR 036)."""

    def is_met(self, start: Pose, command: Command, reading: OdometryReading) -> bool:
        distance = reading.pose.position.distance_to(Point(command.x, command.y))
        return reading.is_at_rest and distance <= EPSILON_D


@dataclass(frozen=True)
class Never:
    """No termination: the step runs until a reaction ends it (V-07)."""

    def is_met(self, start: Pose, command: Command, reading: OdometryReading) -> bool:
        return False


@dataclass(frozen=True)
class Cleared:
    """The code is no longer active (ADR 037); tested on each State, where the codes change."""

    code: Code


PoseTermination = DistanceFromStart | HeadingReached | PositionReached | Never
Termination = PoseTermination | Cleared


@dataclass(frozen=True)
class Twist:
    linear: float  # m/s
    angular: float  # rad/s

    default_termination = None  # a TWIST step names its termination

    def resolve(self, pose: Pose, get_map_to_odom: MapToOdom = get_no_map_to_odom) -> Command:
        return Command(Behavior.TWIST, linear=self.linear, angular=self.angular)


@dataclass(frozen=True)
class TurnTo:
    heading: HeadingTo

    default_termination = HeadingReached()

    def resolve(self, pose: Pose, get_map_to_odom: MapToOdom = get_no_map_to_odom) -> Command:
        heading = self.heading.resolve_heading(pose, get_map_to_odom)
        return Command(Behavior.TURN_TO, theta=wrap_angle(heading))


@dataclass(frozen=True)
class GoTo:
    target: Target

    default_termination = PositionReached()

    def resolve(self, pose: Pose, get_map_to_odom: MapToOdom = get_no_map_to_odom) -> Command:
        point = self.target.resolve_point(pose, get_map_to_odom)
        return Command(Behavior.GO_TO, x=point.x, y=point.y)


StepBehavior = Twist | TurnTo | GoTo


@dataclass(frozen=True)
class Step:
    behavior: StepBehavior
    until: Termination | None = None  # None: the default termination of the behavior
    reactions: Mapping[Code, Reaction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.until is None and self.behavior.default_termination is None:
            raise ValueError(f'{self.behavior} has no default termination: name it with until')
        check_abort_only(self.reactions)

    @property
    def termination(self) -> Termination:
        termination = self.until or self.behavior.default_termination
        assert termination is not None  # checked at construction
        return termination


def wait_until_clear(code: Code) -> Step:
    """Return a step that holds TWIST(0, 0) until the code clears, ignoring it (ADR 038)."""
    # The timeout T_WAIT of ADR 038 comes with the mission by plan (V-04), its only user.
    return Step(Twist(0.0, 0.0), until=Cleared(code), reactions={code: Ignore()})


@dataclass(frozen=True)
class Mission:
    steps: Sequence[Step]
    reactions: Mapping[Code, Reaction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        check_abort_only(self.reactions)

    def get_reaction(self, step: Step, code: Code) -> Reaction:
        """Return the reaction of the step to the code, else of the mission, else abort."""
        return step.reactions.get(code) or self.reactions.get(code) or Abort()
