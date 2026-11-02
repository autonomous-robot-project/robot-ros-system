"""
Closed-loop run on the bench (v_min and T_vel of V-02), recorded in a ros2 bag.

Criticality: T2, product (robot-docs codigo.md).

Publishes the schedule of measure_closed_loop and records the bag that it judges.

Usage, on the RPi with the wheels suspended: run_closed_loop <bag directory>.
"""

import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so the agent and the recorder inherit it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

import sys  # noqa: E402
import time  # noqa: E402

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.publisher import Publisher  # noqa: E402
from robot_interfaces.msg import Command  # noqa: E402

from robot_verification.link import LINK_QOS, BagRecorder, MicroRosAgent  # noqa: E402
from robot_verification.measure_closed_loop import SCHEDULE  # noqa: E402
from robot_verification.recording import (  # noqa: E402
    COMMAND_TOPIC,
    INSTRUMENTATION_TOPIC,
    STATE_TOPIC,
)

RECORDED_TOPICS = [COMMAND_TOPIC, STATE_TOPIC, INSTRUMENTATION_TOPIC]
COMMAND_PERIOD_S = 0.05  # 20 Hz
SESSION_UP_S = 5.0  # agent and recorder up, session of the ESP32 established


def run_schedule(node: Node, publisher: Publisher) -> None:
    for linear, duration_s in SCHEDULE:
        print(f'[{time.strftime("%H:%M:%S")}] TWIST({linear}, 0) for {duration_s} s', flush=True)
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            publisher.publish(Command(behavior=Command.TWIST, linear=linear))
            rclpy.spin_once(node, timeout_sec=COMMAND_PERIOD_S)


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: run_closed_loop <bag directory>')
        sys.exit(2)
    rclpy.init()
    try:
        with MicroRosAgent(), BagRecorder(sys.argv[1], RECORDED_TOPICS):
            time.sleep(SESSION_UP_S)
            node = rclpy.create_node('run_closed_loop')
            run_schedule(node, node.create_publisher(Command, COMMAND_TOPIC, LINK_QOS))
            node.destroy_node()
    finally:
        # The SIGINT handler of rclpy may have shut the context down already.
        rclpy.try_shutdown()
    print(f'done: measure_closed_loop {sys.argv[1]}')


if __name__ == '__main__':
    main()
