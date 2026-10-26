"""The link to the controller, shared by the bench scripts: its QoS and the micro-ROS agent."""

import os
import subprocess

from ament_index_python.packages import get_package_prefix
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# QoS of every topic of the link (ADR 025).
LINK_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# The agent executable is started directly, not through `ros2 run`, which does not forward
# SIGTERM to its child.
AGENT_EXECUTABLE = os.path.join(
    get_package_prefix('micro_ros_agent'), 'lib', 'micro_ros_agent', 'micro_ros_agent'
)
AGENT_ARGUMENTS = ['serial', '--dev', '/dev/ttyAMA0', '-b', '921600']  # UART of ADR 043


class MicroRosAgent:
    """The micro-ROS agent on the UART of the link, as a child process."""

    def __init__(self):
        self._process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exception):
        self.stop()

    def start(self):
        self._process = subprocess.Popen(
            [AGENT_EXECUTABLE, *AGENT_ARGUMENTS],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self):
        """Stop the agent if it is still running; safe to call more than once."""
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            self._process.communicate()

    def exit_report(self):
        """Return why the agent exited, or None while it runs."""
        if self._process.poll() is None:
            return None
        _, stderr = self._process.communicate()
        return f'agent exited with code {self._process.returncode}: {stderr.strip()}'
