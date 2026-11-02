"""
Node of the sequencer: translates the topics of the controller into calls to the Sequencer.

Criticality: T2, product (robot-docs codigo.md).

One node, one thread (ADRs 031 and 033): the State (100 Hz) and the Odometry (20 Hz) of the
controller and the heartbeat of the notebook go to the Sequencer, a command it returns is
published in the same callback, and a timer checks the ages of the State and of the heartbeat
(ADR 037) and republishes the command in force at f_cmd so it does not expire (RNF-02). The times
come from the steady clock, so an adjustment of the system clock does not age a message. With no
mission running the command is TWIST(0, 0). The mission comes from the parameter `mission`
(arquitetura.md). The node holds no logic of its own: the domain is in sequencer.py and
mission.py (ADR 047).

Usage: chrt -f 50 <prefix>/lib/robot_sequencer/sequencer --ros-args -p mission:=v01
(the executable itself, not `ros2 run`, which does not forward SIGTERM; priority of ADR 039).
"""

import math

from nav_msgs.msg import Odometry
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robot_interfaces.msg import Command as CommandMessage
from robot_interfaces.msg import State
from std_msgs.msg import Empty

from robot_sequencer.geometry import Point, Pose
from robot_sequencer.mission import Behavior, Code, Command, OdometryReading, StateReading
from robot_sequencer.missions import MISSIONS
from robot_sequencer.sequencer import Sequencer

COMMAND_TOPIC = '/controller/command'
STATE_TOPIC = '/controller/state'
ODOMETRY_TOPIC = '/controller/odometry'
HEARTBEAT_TOPIC = '/heartbeat'  # std_msgs/Empty at 10 Hz from the notebook (arquitetura.md)

# QoS of every topic of the link (ADR 025), the same as robot_verification/link.py.
LINK_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

COMMAND_PERIOD_S: float = 0.05  # 1 / f_cmd, f_cmd = 20 Hz (ADR 033)
NS_PER_S: float = 1e9

FAULT_CODES: dict[int, Code] = {
    State.FAULT_COGNIZANT: Code.FAULT_COGNIZANT,
    State.FAULT_DEADLINE_MISSED: Code.FAULT_DEADLINE_MISSED,
    State.FAULT_WATCHDOG_RESTART: Code.FAULT_WATCHDOG_RESTART,
    State.FAULT_FRONT_SENSOR: Code.FAULT_FRONT_SENSOR,
    State.FAULT_TEST: Code.FAULT_TEST,
}
REFLEX_CODES: dict[int, Code] = {
    State.REFLEX_FRONT_OBSTACLE: Code.REFLEX_FRONT_OBSTACLE,
}
MASK_BITS = 32  # uint32 masks of State.msg


def get_active_codes(state: State) -> frozenset[Code]:
    return get_codes_of_mask(state.faults, FAULT_CODES) | get_codes_of_mask(
        state.reflexes, REFLEX_CODES
    )


def get_codes_of_mask(mask: int, codes: dict[int, Code]) -> frozenset[Code]:
    """Return the code of each set bit; a bit not in `codes` is Code.UNKNOWN."""
    bits = (1 << position for position in range(MASK_BITS))
    return frozenset(codes.get(bit, Code.UNKNOWN) for bit in bits if mask & bit)


def get_command(message: CommandMessage) -> Command:
    return Command(
        Behavior(message.behavior),
        linear=message.linear,
        angular=message.angular,
        theta=message.theta,
        x=message.x,
        y=message.y,
    )


def get_state_reading(state: State) -> StateReading:
    return StateReading(get_active_codes(state), echo=get_command(state.echo))


def get_odometry_reading(odometry: Odometry) -> OdometryReading:
    position = odometry.pose.pose.position
    orientation = odometry.pose.pose.orientation
    # Planar pose: the quaternion is a rotation about z only.
    theta = 2.0 * math.atan2(orientation.z, orientation.w)
    return OdometryReading(
        Pose(Point(position.x, position.y), theta),
        linear=odometry.twist.twist.linear.x,
        angular=odometry.twist.twist.angular.z,
    )


def build_command_message(command: Command) -> CommandMessage:
    return CommandMessage(
        behavior=command.behavior.value,
        linear=command.linear,
        angular=command.angular,
        theta=command.theta,
        x=command.x,
        y=command.y,
    )


class SequencerNode(Node):
    def __init__(self):
        super().__init__('sequencer')
        mission_name = self.declare_parameter('mission', '').value
        if mission_name not in MISSIONS:
            # The mission by plan (empty `mission`) comes with V-04.
            raise ValueError(f'mission {mission_name!r} unknown; use one of {sorted(MISSIONS)}')
        self._sequencer = Sequencer(MISSIONS[mission_name]())
        self._commands = self.create_publisher(CommandMessage, COMMAND_TOPIC, LINK_QOS)
        self.create_subscription(State, STATE_TOPIC, self._on_state, LINK_QOS)
        self.create_subscription(Odometry, ODOMETRY_TOPIC, self._on_odometry, LINK_QOS)
        self.create_subscription(Empty, HEARTBEAT_TOPIC, self._on_heartbeat, LINK_QOS)
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(COMMAND_PERIOD_S, self._on_timer)
        self.get_logger().info(f'mission {mission_name}: waiting for the first Odometry')

    def _on_state(self, state: State) -> None:
        self._publish_if_new(self._sequencer.on_state(get_state_reading(state), self._get_now_s()))

    def _on_odometry(self, odometry: Odometry) -> None:
        self._publish_if_new(self._sequencer.on_odometry(get_odometry_reading(odometry)))

    def _on_heartbeat(self, _: Empty) -> None:
        self._sequencer.on_heartbeat(self._get_now_s())

    def _on_timer(self) -> None:
        """Publish the command in force, the new one if an age activated a code."""
        self._log_if_new(self._sequencer.on_timer(self._get_now_s()))
        self._commands.publish(build_command_message(self._sequencer.command))

    def _publish_if_new(self, command: Command | None) -> None:
        if command is None:
            return
        self._commands.publish(build_command_message(command))
        self._log_if_new(command)

    def _log_if_new(self, command: Command | None) -> None:
        if command is not None:
            self.get_logger().info(f'{self._sequencer.status.name}: {command}')

    def _get_now_s(self) -> float:
        return self._steady_clock.now().nanoseconds / NS_PER_S


def main() -> None:
    rclpy.init()
    try:
        rclpy.spin(SequencerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
