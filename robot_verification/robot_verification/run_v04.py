"""
V-04 procedure on the bench: conducts the trial and records the ros2 bag that check_v04 judges.

Criticality: T2, product (robot-docs codigo.md).

Steps of V-04: load M in the deliberator with the pose at p_0 (trial launch of robot_bringup:
map_server, planner_server, map -> odom static, odom_to_tf_ros2), with the heartbeat of the
notebook active and the front of the VL53L0X clear beyond d_stop; record the plan, the commands of
the sequencer, the State and the Odometry; ask the sequencer for a plan to p_g; wait for
TWIST(0, 0) and stop the recording.

The plan is asked for as in operation (F7 D4): the sequencer runs with an empty mission and
receives /goal_pose once, published after it subscribes. The heartbeat comes from the notebook
(`ros2 topic pub -r 10 /heartbeat std_msgs/msg/Empty`, robot-docs arquitetura.md); the script only
checks that it has a publisher before asking for the plan. The static transforms are recorded, so
that check_v04 can confirm that map -> odom is the identity.

Usage, on the RPi with the wheels suspended and the motor switch on: run_v04 <bag directory>.
"""

from collections.abc import Callable
import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so the children inherit it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from geometry_msgs.msg import PoseStamped  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from robot_interfaces.msg import Command, State  # noqa: E402

from robot_verification.check_v04 import P_G, PLAN_TOPIC, TF_STATIC_TOPIC  # noqa: E402
from robot_verification.link import LINK_QOS, BagRecorder, MicroRosAgent  # noqa: E402
from robot_verification.recording import (  # noqa: E402
    COMMAND_TOPIC,
    ODOMETRY_TOPIC,
    STATE_TOPIC,
)
from robot_verification.run_v07 import SequencerProcess  # noqa: E402

RECORDED_TOPICS: list[str] = [
    PLAN_TOPIC,
    TF_STATIC_TOPIC,
    COMMAND_TOPIC,
    STATE_TOPIC,
    ODOMETRY_TOPIC,
]
GOAL_TOPIC = '/goal_pose'
HEARTBEAT_TOPIC = '/heartbeat'
DELIBERATOR_LAUNCH: list[str] = ['ros2', 'launch', 'robot_bringup', 'deliberator_test.launch.py']
MISSION_BY_PLAN: str = ''  # an empty mission waits for /goal_pose
SESSION_UP_S: float = 5.0  # agent, recorder, deliberator and sequencer up
WAIT_FOR_PEER_S: float = 30.0  # heartbeat publisher and goal subscriber
# The final TWIST(0, 0) must hold this long after the last GO_TO: beyond the 20 Hz republication
# and the stop of the wheels.
STOP_HOLD_S: float = 1.0
# The four segments of M at v_max take well under a minute; beyond this the run is stopped and
# check_v04 judges what was recorded.
MAX_TRIAL_S: float = 120.0
POLL_S: float = 0.1
LAUNCH_STOP_S: float = 10.0


class DeliberatorLaunch:
    """The trial launch of the deliberator, as a child process."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> 'DeliberatorLaunch':
        self._process = subprocess.Popen(DELIBERATOR_LAUNCH)
        return self

    def __exit__(self, *_exception: object) -> None:
        # SIGINT, not SIGTERM: ros2 launch stopped by SIGTERM left its nodes running
        # (robot-docs nota 0003).
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=LAUNCH_STOP_S)
        except subprocess.TimeoutExpired:
            self._process.kill()


class TrialNode:
    """Publishes the goal and follows the echo of the State to know when the run is over."""

    def __init__(self) -> None:
        self._node = rclpy.create_node('run_v04')
        self._goals = self._node.create_publisher(PoseStamped, GOAL_TOPIC, 10)
        self._node.create_subscription(State, STATE_TOPIC, self._on_state, LINK_QOS)
        self._saw_go_to: bool = False
        self._stop_since_s: float | None = None
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self) -> 'TrialNode':
        self._spinning.start()
        return self

    def __exit__(self, *_exception: object) -> None:
        self._executor.shutdown()
        self._node.destroy_node()

    def has_heartbeat(self) -> bool:
        return self._node.count_publishers(HEARTBEAT_TOPIC) > 0

    def has_goal_subscriber(self) -> bool:
        return self._goals.get_subscription_count() > 0

    def publish_goal(self) -> None:
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = P_G
        goal.pose.orientation.w = 1.0
        self._goals.publish(goal)

    def has_stopped_after_the_plan(self) -> bool:
        """Tell whether the echo is TWIST(0, 0) for STOP_HOLD_S after a GO_TO."""
        if not self._saw_go_to or self._stop_since_s is None:
            return False
        return time.monotonic() - self._stop_since_s >= STOP_HOLD_S

    def _on_state(self, state: State) -> None:
        echo = state.echo
        is_stop = echo.behavior == Command.TWIST and echo.linear == 0.0 and echo.angular == 0.0
        self._saw_go_to = self._saw_go_to or echo.behavior == Command.GO_TO
        if not is_stop:
            self._stop_since_s = None
        elif self._stop_since_s is None:
            self._stop_since_s = time.monotonic()


class V04Procedure:
    """The steps of V-04, announced on the terminal."""

    def __init__(self, trial: TrialNode):
        self._trial = trial

    def run(self) -> None:
        announce('on the notebook: ros2 topic pub -r 10 /heartbeat std_msgs/msg/Empty')
        if not wait_until(self._trial.has_heartbeat, WAIT_FOR_PEER_S):
            announce('no heartbeat publisher: stopping without asking for the plan')
            return
        if not wait_until(self._trial.has_goal_subscriber, WAIT_FOR_PEER_S):
            announce('the sequencer does not subscribe to /goal_pose: stopping')
            return
        self._trial.publish_goal()
        announce(f'goal p_g = {P_G} published; waiting for TWIST(0, 0) after the plan')
        if not wait_until(self._trial.has_stopped_after_the_plan, MAX_TRIAL_S):
            announce(f'no final TWIST(0, 0) within {MAX_TRIAL_S:.0f} s: stopping')


def wait_until(condition: Callable[[], bool], timeout_s: float) -> bool:
    """Poll the condition until it holds or the timeout ends; return whether it held."""
    deadline_s = time.monotonic() + timeout_s
    while time.monotonic() < deadline_s:
        if condition():
            return True
        time.sleep(POLL_S)
    return False


def announce(text: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: run_v04 <bag directory>')
        sys.exit(2)
    rclpy.init()
    try:
        with (
            MicroRosAgent(),
            BagRecorder(sys.argv[1], RECORDED_TOPICS),
            DeliberatorLaunch(),
            SequencerProcess(MISSION_BY_PLAN),
            TrialNode() as trial,
        ):
            announce('V-04: agent, recorder, deliberator and sequencer up; wheels suspended?')
            time.sleep(SESSION_UP_S)
            V04Procedure(trial).run()
    finally:
        rclpy.shutdown()
    announce(f'done: check_v04 {sys.argv[1]}')


if __name__ == '__main__':
    main()
