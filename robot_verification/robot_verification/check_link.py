"""
Bench check of the link: the controller state arrives, and arrives again after an agent restart.

Criticality: T2, product (robot-docs codigo.md).

Run at the start of each bench session and after any ROS update. A crash of the agent on the
first message from the ESP32 is the symptom of an agent built against older ROS libraries
(ADR 043, robot-docs nota 0005). The restart exercises the session rebuild of the ESP32.
"""

import sys
import time

import rclpy
from robot_interfaces.msg import State

from robot_verification.link import LINK_QOS, MicroRosAgent

STATE_TOPIC = '/controller/state'
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
        node.create_subscription(State, STATE_TOPIC, self._on_state, LINK_QOS)

    def _on_state(self, _message):
        self.times.append(time.monotonic())

    def any_since(self, instant):
        return bool(self.times) and self.times[-1] > instant


def wait_for_state(node, arrivals, agent, phase):
    """Wait for a state that arrived after the start of the phase; return the seconds."""
    start = time.monotonic()
    while time.monotonic() - start < TIMEOUT_S:
        exit_report = agent.exit_report()
        if exit_report is not None:
            raise CheckError(f'{phase}: {exit_report}')
        rclpy.spin_once(node, timeout_sec=POLL_S)
        if arrivals.any_since(start):
            return time.monotonic() - start
    raise CheckError(f'{phase}: no state within {TIMEOUT_S:g} s')


def run_check(node):
    arrivals = StateArrivals(node)
    agent = MicroRosAgent()
    try:
        agent.start()
        first_s = wait_for_state(node, arrivals, agent, 'first connection')
        agent.stop()
        time.sleep(AGENT_DOWN_S)
        agent.start()
        again_s = wait_for_state(node, arrivals, agent, 'reconnection')
    finally:
        agent.stop()
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
