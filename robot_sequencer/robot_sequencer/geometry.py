"""
Planar geometry of the sequencer: a point and a pose in odom or map, and the wrap of an angle.

Criticality: T2, product (robot-docs codigo.md).

Domain without ROS (ADR 047): the node converts the Odometry to a Pose, so the targets and the
terminations of the steps are tested without ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def wrap_angle(angle: float) -> float:
    """Return the same angle in [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Point:
    x: float  # m, in odom (in map for a vertex of the plan)
    y: float  # m, in odom (in map for a vertex of the plan)

    def distance_to(self, other: Point) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def direction_to(self, other: Point) -> float:
        """Return the heading in odom (rad) of the line from this point to the other."""
        return math.atan2(other.y - self.y, other.x - self.x)


@dataclass(frozen=True)
class Pose:
    position: Point
    theta: float  # rad, heading in odom (in map for the pose of odom)

    def convert_to_local(self, point: Point) -> Point:
        """
        Return in the frame of this pose a point given in its parent frame.

        For the map -> odom transform, the pose is odom in map: a vertex in map comes back in odom.
        """
        dx = point.x - self.position.x
        dy = point.y - self.position.y
        cos, sin = math.cos(self.theta), math.sin(self.theta)
        return Point(cos * dx + sin * dy, -sin * dx + cos * dy)
