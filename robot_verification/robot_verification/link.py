"""
The link to the controller, shared by the bench scripts: QoS, micro-ROS agent, bag recorder.

Criticality: T2, product (robot-docs codigo.md).
"""

import os
import signal
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
# Real-time priority of the agent and the sequencer (ADR 039): `chrt` replaces itself with the
# agent, so SIGTERM still reaches it. Needs `rtprio 50` in limits.d on the RPi.
REALTIME_PREFIX = ['chrt', '-f', '50']


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
            [*REALTIME_PREFIX, AGENT_EXECUTABLE, *AGENT_ARGUMENTS],
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


class SequencerProcess:
    """The node of the sequencer running a mission, as a child process, its log in a file."""

    def __init__(self, mission, log_path):
        executable = os.path.join(
            get_package_prefix('robot_sequencer'), 'lib', 'robot_sequencer', 'sequencer'
        )
        self._command = [*REALTIME_PREFIX, executable, '--ros-args', '-p', f'mission:={mission}']
        self._log_path = log_path
        self._process = None

    def __enter__(self):
        with open(self._log_path, 'w') as log:
            self._process = subprocess.Popen(self._command, stdout=log, stderr=subprocess.STDOUT)
        return self

    def __exit__(self, *_exception):
        # The executable itself, not `ros2 run`: SIGTERM reaches the node, which exits cleanly.
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()


class BagRecorder:
    """`ros2 bag record` of the given topics, as a child process."""

    def __init__(self, path, topics):
        # ros2 bag refuses an existing directory, and its error would be lost with stderr.
        if os.path.exists(path):
            raise FileExistsError(f'bag directory already exists: {path}')
        self._command = ['ros2', 'bag', 'record', '-s', 'mcap', '-o', path, *topics]
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
