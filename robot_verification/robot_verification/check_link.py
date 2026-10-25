"""
Bench check of the link: the agent starts and the controller joins the ROS graph (ADR 043).

Run at the start of each bench session and after any ROS update. A crash of the agent on the
first message from the ESP32 is the symptom of an agent built against older ROS libraries.
"""

import os
import subprocess
import sys
import time

from ament_index_python.packages import get_package_prefix
import rclpy

# The agent executable is started directly, not through `ros2 run`, which does not forward
# SIGTERM to its child.
AGENT_EXECUTABLE = os.path.join(
    get_package_prefix('micro_ros_agent'), 'lib', 'micro_ros_agent', 'micro_ros_agent'
)
AGENT_ARGUMENTS = ['serial', '--dev', '/dev/ttyAMA0', '-b', '921600']
CONTROLLER_NODE = 'controller'
TIMEOUT_S = 10.0
POLL_S = 0.5


def wait_for_controller(node, agent):
    """Return the seconds until the controller joins, or None on timeout or agent exit."""
    start = time.monotonic()
    while time.monotonic() - start < TIMEOUT_S and agent.poll() is None:
        rclpy.spin_once(node, timeout_sec=POLL_S)
        if CONTROLLER_NODE in node.get_node_names():
            return time.monotonic() - start
    return None


def main():
    agent = subprocess.Popen(
        [AGENT_EXECUTABLE, *AGENT_ARGUMENTS],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    rclpy.init()
    node = rclpy.create_node('check_link')
    try:
        joined_after_s = wait_for_controller(node, agent)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        agent_exit_code = agent.poll()
        if agent_exit_code is None:
            agent.terminate()
        _, agent_stderr = agent.communicate()

    if agent_exit_code is not None:
        print(f'FAIL: agent exited with code {agent_exit_code}: {agent_stderr.strip()}')
        sys.exit(1)
    if joined_after_s is None:
        print(f'FAIL: /{CONTROLLER_NODE} not seen within {TIMEOUT_S:g} s')
        sys.exit(1)
    print(f'PASS: /{CONTROLLER_NODE} joined after {joined_after_s:.1f} s')


if __name__ == '__main__':
    main()
