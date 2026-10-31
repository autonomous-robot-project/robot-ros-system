"""
V-03 procedure on the bench: conducts the trial and records the ros2 bag that check_v03 judges.

Criticality: T2, product (robot-docs codigo.md).

Steps of V-03: record the command, the Odometry and the State; TWIST(v_r, 0) at 20 Hz, with the
front of the sensor clear beyond d_stop, to steady state; place a flat screen at d_o in front of
the VL53L0X, hold it for t_o and remove it, three times; stop the recording.

Each placement and removal is announced on the terminal at the end of a countdown and marked in the
bag (/verification/v03_marker), because check_v03 judges the speed against these windows and not
against the reflex bit, which the controller itself sets (robot-docs nota 0015, D8).

Usage, on the RPi with the wheels suspended: run_v03 <bag directory>.
"""

import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so the agent and the recorder inherit it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from robot_interfaces.msg import Command  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from robot_verification.check_v03 import (  # noqa: E402
    EXPECTED_SCREENS,
    MARKER_TOPIC,
    PLACED,
    REMOVED,
    T_O_S,
    V_R,
)
from robot_verification.link import LINK_QOS, BagRecorder, MicroRosAgent  # noqa: E402
from robot_verification.recording import (  # noqa: E402
    COMMAND_TOPIC,
    ODOMETRY_TOPIC,
    STATE_TOPIC,
)

D_O_M: float = 0.10  # d_o, distance of the screen from the sensor (V-03)
COMMAND_PERIOD_S: float = 0.05  # 20 Hz, above 1/T_cmd
RECORDED_TOPICS: list[str] = [COMMAND_TOPIC, STATE_TOPIC, ODOMETRY_TOPIC, MARKER_TOPIC]
SESSION_UP_S: float = 5.0  # agent and recorder up, session of the ESP32 established
STEADY_STATE_S: float = 5.0  # beyond the ramp and the settling; above T_resume + 1 s of check_v03
COUNTDOWN_S: int = 3


class TrialNode:
    """Publishes TWIST(v_r, 0) at 20 Hz and the markers of the procedure."""

    def __init__(self) -> None:
        self._node = rclpy.create_node('run_v03')
        self._commands = self._node.create_publisher(Command, COMMAND_TOPIC, LINK_QOS)
        self._markers = self._node.create_publisher(String, MARKER_TOPIC, 10)
        self._command = Command(behavior=Command.TWIST, linear=V_R)
        self._node.create_timer(COMMAND_PERIOD_S, self._publish_command)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self) -> 'TrialNode':
        self._spinning.start()
        return self

    def __exit__(self, *_exception: object) -> None:
        self._executor.shutdown()
        self._node.destroy_node()

    def mark(self, name: str) -> None:
        self._markers.publish(String(data=name))

    def _publish_command(self) -> None:
        self._commands.publish(self._command)


class V03Procedure:
    """The steps of V-03, announced on the terminal and marked in the bag."""

    def __init__(self, trial: TrialNode):
        self._trial = trial

    def run(self) -> None:
        announce(f'TWIST({V_R}, 0): keep the front of the sensor clear; wait for the steady state')
        time.sleep(STEADY_STATE_S)
        for number in range(1, EXPECTED_SCREENS + 1):
            self._screen(number)

    def _screen(self, number: int) -> None:
        countdown(f'screen {number} of {EXPECTED_SCREENS}: place it at {D_O_M} m from the sensor')
        self._trial.mark(PLACED)
        announce('PLACE NOW')
        # The hold is the countdown to the removal, so the operator never removes early.
        countdown('hold it; remove it only at NOW', round(T_O_S))
        self._trial.mark(REMOVED)
        announce('REMOVE NOW')
        time.sleep(STEADY_STATE_S)


def announce(text: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def countdown(text: str, seconds: int = COUNTDOWN_S) -> None:
    announce(text)
    for remaining in range(seconds, 0, -1):
        announce(f'  {remaining}...')
        time.sleep(1.0)


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: run_v03 <bag directory>')
        sys.exit(2)
    rclpy.init()
    try:
        with MicroRosAgent(), BagRecorder(sys.argv[1], RECORDED_TOPICS), TrialNode() as trial:
            announce('V-03: agent and recorder up; wheels suspended?')
            time.sleep(SESSION_UP_S)
            V03Procedure(trial).run()
    finally:
        rclpy.shutdown()
    announce(f'done: check_v03 {sys.argv[1]}')


if __name__ == '__main__':
    main()
