"""
Vocabulary of the missions: steps of primitive behaviors, their targets and terminations.

Criticality: T2, product (robot-docs codigo.md).

A mission is data (ADR 032): a list of steps, each with a behavior of ADR 024, parameters computed
at the entry of the step from the current pose, and a termination tested on each Odometry
(ADR 033). The reactions to the codes of the controller are tables per step and per mission; the
missions of V-01 leave them empty, so every activation of a code aborts, the default.
Domain without ROS (ADR 047): the node converts the messages to these types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
import math

from robot_sequencer.geometry import Point, Pose, wrap_angle

# Tolerances of the default terminations (V-01): above the worst final error of the controller
# plus the resolution of the odometry, measured on the bench (robot-docs nota 0014):
# 3.6 deg + 3.9 deg per edge of one wheel, and 6 mm + 10 mm per pulse.
EPSILON_THETA: float = math.radians(10.0)  # rad, ε_θ
EPSILON_D: float = 0.05  # m, ε_d


class Behavior(Enum):
    """Primitive behavior of the controller, with the values of Command.msg (ADR 024)."""

    TWIST = 0
    TURN_TO = 1
    GO_TO = 2


@dataclass(frozen=True)
class Command:
    """Behavior and parameters sent to the controller; the unused fields are zero."""

    behavior: Behavior
    linear: float = 0.0  # m/s, TWIST
    angular: float = 0.0  # rad/s, TWIST
    theta: float = 0.0  # rad in odom, TURN_TO
    x: float = 0.0  # m in odom, GO_TO
    y: float = 0.0  # m in odom, GO_TO


# TWIST(0, 0): the command when no mission runs, after its end and after an abort (ADR 030).
STOP = Command(Behavior.TWIST)


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
    """Condition reported by the controller, one bit of a mask of the State (ADR 025)."""

    FAULT_COGNIZANT = auto()
    FAULT_DEADLINE_MISSED = auto()
    FAULT_WATCHDOG_RESTART = auto()
    FAULT_FRONT_SENSOR = auto()
    FAULT_TEST = auto()
    REFLEX_FRONT_OBSTACLE = auto()
    UNKNOWN = auto()  # a bit the sequencer does not know yet: the default reaction still applies


@dataclass(frozen=True)
class Abort:
    """Reaction that ends the mission with TWIST(0, 0), the default for every code (ADR 032)."""


Reaction = Abort


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

    def resolve_point(self, pose: Pose) -> Point:
        if self._point is None:
            heading = pose.theta + self._turn
            self._point = Point(
                pose.position.x + self._distance * math.cos(heading),
                pose.position.y + self._distance * math.sin(heading),
            )
        return self._point


@dataclass(frozen=True)
class HeadingTo:
    """Heading from the current pose to a point target."""

    target: Offset

    def resolve_heading(self, pose: Pose) -> float:
        return pose.position.direction_to(self.target.resolve_point(pose))


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


Termination = DistanceFromStart | HeadingReached | PositionReached


@dataclass(frozen=True)
class Twist:
    linear: float  # m/s
    angular: float  # rad/s

    default_termination = None  # a TWIST step names its termination

    def resolve(self, pose: Pose) -> Command:
        return Command(Behavior.TWIST, linear=self.linear, angular=self.angular)


@dataclass(frozen=True)
class TurnTo:
    heading: HeadingTo

    default_termination = HeadingReached()

    def resolve(self, pose: Pose) -> Command:
        return Command(Behavior.TURN_TO, theta=wrap_angle(self.heading.resolve_heading(pose)))


@dataclass(frozen=True)
class GoTo:
    target: Offset

    default_termination = PositionReached()

    def resolve(self, pose: Pose) -> Command:
        point = self.target.resolve_point(pose)
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

    @property
    def termination(self) -> Termination:
        termination = self.until or self.behavior.default_termination
        assert termination is not None  # checked at construction
        return termination


@dataclass(frozen=True)
class Mission:
    steps: Sequence[Step]
    reactions: Mapping[Code, Reaction] = field(default_factory=dict)
