"""
V-02 procedure on the bench: conducts the trial and records the ros2 bag that check_v02 judges.

Steps of V-02: record the State; TWIST(v_r, 0) to steady state; cut the supply (VM) of the
motors with the switch for 3 s and restore it, three times; touch GPIO 11 to GND for an instant
(watchdog test trigger) and wait for the restart; stop the recording.

Each cut, restore and injection is announced on the terminal and marked in the bag
(/verification/v02_marker), because check_v02 judges the failure code against these windows and
not against the measured speed, which the controller itself uses.

Usage, on the RPi with the wheels suspended: run_v02 <bag directory>.
"""

import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so the agent and the recorder inherit it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from robot_interfaces.msg import Command  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from robot_verification.check_v02 import (  # noqa: E402
    CUT,
    EXPECTED_CUTS,
    INJECT,
    MARKER_TOPIC,
    RESTORE,
    V_R,
)
from robot_verification.link import LINK_QOS, MicroRosAgent  # noqa: E402

COMMAND_TOPIC = '/controller/command'
COMMAND_PERIOD_S = 0.05  # 20 Hz
RECORDED_TOPICS = [
    '/controller/state',
    '/controller/command',
    '/controller/instrumentation',
    MARKER_TOPIC,
]
SESSION_UP_S = 5.0  # agent and recorder up, session of the ESP32 established
STEADY_STATE_S = 5.0  # V-02 step 2, well beyond the settling time
CUT_S = 3.0  # V-02 step 3
COUNTDOWN_S = 3
RESTART_WAIT_S = 15.0  # restart, reconnection of the session and the code of the watchdog


class TrialNode:
    """Publishes TWIST(v_r, 0) at 20 Hz and the markers of the procedure."""

    def __init__(self):
        self._node = rclpy.create_node('run_v02')
        self._commands = self._node.create_publisher(Command, COMMAND_TOPIC, LINK_QOS)
        self._markers = self._node.create_publisher(String, MARKER_TOPIC, 10)
        self._command = Command(behavior=Command.TWIST, linear=V_R)
        self._node.create_timer(COMMAND_PERIOD_S, self._publish_command)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self):
        self._spinning.start()
        return self

    def __exit__(self, *_exception):
        self._executor.shutdown()
        self._node.destroy_node()

    def mark(self, name):
        self._markers.publish(String(data=name))

    def _publish_command(self):
        self._commands.publish(self._command)


class BagRecorder:
    """`ros2 bag record` of the topics of the trial, as a child process."""

    def __init__(self, path):
        self._command = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', path, *RECORDED_TOPICS]
        self._process = None

    def __enter__(self):
        self._process = subprocess.Popen(
            self._command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return self

    def __exit__(self, *_exception):
        # SIGINT closes the bag properly; the recorder is a direct child, so it receives it.
        self._process.send_signal(signal.SIGINT)
        self._process.wait()


class V02Procedure:
    """The steps of V-02, announced on the terminal and marked in the bag."""

    def __init__(self, trial):
        self._trial = trial

    def run(self):
        announce(f'TWIST({V_R}, 0): wait for the steady state')
        time.sleep(STEADY_STATE_S)
        for number in range(1, EXPECTED_CUTS + 1):
            self._cut(number)
        self._inject()

    def _cut(self, number):
        countdown(f'cut {number} of {EXPECTED_CUTS}: switch OFF the supply of the motors')
        self._trial.mark(CUT)
        announce('OFF NOW')
        # The end of the cut is the countdown to restore it, so the operator never restores early.
        time.sleep(CUT_S - COUNTDOWN_S)
        countdown('switch it back ON only at NOW')
        self._trial.mark(RESTORE)
        announce('ON NOW')
        time.sleep(STEADY_STATE_S)

    def _inject(self):
        countdown('watchdog: touch GPIO 11 to GND for an instant')
        self._trial.mark(INJECT)
        announce('TOUCH NOW')
        time.sleep(RESTART_WAIT_S)


def announce(text):
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def countdown(text):
    announce(text)
    for remaining in range(COUNTDOWN_S, 0, -1):
        announce(f'  {remaining}...')
        time.sleep(1.0)


def main():
    if len(sys.argv) != 2:
        print('usage: run_v02 <bag directory>')
        sys.exit(2)
    rclpy.init()
    try:
        with MicroRosAgent(), BagRecorder(sys.argv[1]), TrialNode() as trial:
            announce('V-02: agent and recorder up; wheels suspended?')
            time.sleep(SESSION_UP_S)
            V02Procedure(trial).run()
    finally:
        rclpy.shutdown()
    announce(f'done: check_v02 {sys.argv[1]}')


if __name__ == '__main__':
    main()
