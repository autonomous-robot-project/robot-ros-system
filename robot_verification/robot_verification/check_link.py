"""
Bench check of the link: the controller state arrives, and arrives again after an agent restart.

Run at the start of each bench session and after any ROS update. A crash of the agent on the
first message from the ESP32 is the symptom of an agent built against older ROS libraries
(ADR 043, robot-docs nota 0005). The restart exercises the session rebuild of the ESP32.
"""

import os
import subprocess
import sys
import time

from ament_index_python.packages import get_package_prefix
import rclpy
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from robot_interfaces.msg import State

# The agent executable is started directly, not through `ros2 run`, which does not forward
# SIGTERM to its child.
AGENT_EXECUTABLE = os.path.join(
    get_package_prefix('micro_ros_agent'), 'lib', 'micro_ros_agent', 'micro_ros_agent'
)
AGENT_ARGUMENTS = ['serial', '--dev', '/dev/ttyAMA0', '-b', '921600']
STATE_TOPIC = '/controller/state'
# QoS of the link (ADR 025).
STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
TIMEOUT_S = 10.0
POLL_S = 0.1
# Agent down for longer than the ESP32 needs to notice it: ping period (1 s) plus ping timeout.
AGENT_DOWN_S = 3.0


class CheckError(Exception):
    """A phase of the check did not complete."""


class StateArrivals:
    """Arrival times of the controller state."""

    def __init__(self, node):
        self.times = []
        node.create_subscription(State, STATE_TOPIC, self._on_state, STATE_QOS)

    def _on_state(self, _message):
        self.times.append(time.monotonic())

    def any_since(self, instant):
        return bool(self.times) and self.times[-1] > instant


def start_agent():
    return subprocess.Popen(
        [AGENT_EXECUTABLE, *AGENT_ARGUMENTS],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def stop_agent(agent):
    """Stop the agent if it is still running; safe to call more than once."""
    if agent.returncode is None:
        agent.terminate()
        agent.communicate()


def wait_for_state(node, arrivals, agent, phase):
    """Wait for a state that arrived after the start of the phase; return the seconds."""
    start = time.monotonic()
    while time.monotonic() - start < TIMEOUT_S:
        if agent.poll() is not None:
            _, agent_stderr = agent.communicate()
            raise CheckError(
                f'{phase}: agent exited with code {agent.returncode}: {agent_stderr.strip()}'
            )
        rclpy.spin_once(node, timeout_sec=POLL_S)
        if arrivals.any_since(start):
            return time.monotonic() - start
    raise CheckError(f'{phase}: no state within {TIMEOUT_S:g} s')


def run_check(node):
    arrivals = StateArrivals(node)
    agent = start_agent()
    try:
        first_s = wait_for_state(node, arrivals, agent, 'first connection')
        stop_agent(agent)
        time.sleep(AGENT_DOWN_S)
        agent = start_agent()
        again_s = wait_for_state(node, arrivals, agent, 'reconnection')
    finally:
        stop_agent(agent)
    return first_s, again_s


def main():
    rclpy.init()
    node = rclpy.create_node('check_link')
    try:
        first_s, again_s = run_check(node)
    except CheckError as failure:
        print(f'FAIL: {failure}')
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    print(f'PASS: state after {first_s:.1f} s; after the agent restart, after {again_s:.1f} s')


if __name__ == '__main__':
    main()
