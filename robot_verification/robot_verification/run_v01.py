"""
V-01 procedure on the bench: conducts both executions and records what check_v01 judges.

Criticality: T2, product (robot-docs codigo.md).

Steps of V-01: start the system without the nodes of the deliberator, with the front of the
VL53L0X clear beyond d_stop; record each execution in its own bag (command, State, Odometry);
nominal execution: start the sequence and wait for S4, holding S4 for T_obs; execution with the
fault: restart the sequence and, during S1, lock a wheel by hand until the sequencer commands S4,
then release the wheel.

Restarting the sequence is restarting the node of the sequencer (arquitetura.md). The list of
nodes is saved at the start and at the end of each execution, for RF-05 (robot-docs nota 0016).
The command topic is watched only to time the prompts and the end of an execution; the verdict
comes from check_v01.

Usage, on the RPi with the wheels suspended: run_v01 <run directory>. Writes nominal/ and falha/
inside, each with bag/, nodes_antes.txt, nodes_depois.txt and sequencer.log.
"""

import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so every child process inherits it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

from collections.abc import Callable  # noqa: E402
from pathlib import Path  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from robot_interfaces.msg import Command  # noqa: E402

from robot_verification.check_v01 import (  # noqa: E402
    BAG,
    FAILURE,
    NODES_AFTER,
    NODES_BEFORE,
    NOMINAL,
    SEQUENCER_NODE,
    Stage,
    get_stage,
)
from robot_verification.link import (  # noqa: E402
    LINK_QOS,
    BagRecorder,
    MicroRosAgent,
    SequencerProcess,
)
from robot_verification.recording import COMMAND_TOPIC, ODOMETRY_TOPIC, STATE_TOPIC  # noqa: E402

MISSION: str = 'v01'
T_OBS_S: float = 5.0  # T_obs, S4 held after the stop (V-01)
RECORDED_TOPICS: list[str] = [COMMAND_TOPIC, STATE_TOPIC, ODOMETRY_TOPIC]
SESSION_UP_S: float = 5.0  # agent up, session of the ESP32 established
RECORDER_UP_S: float = 2.0  # the recorder discovers the topics before the sequencer starts
NODE_DISCOVERY_TIMEOUT_S: float = 10.0  # for the sequencer to appear in `ros2 node list`
EXECUTION_TIMEOUT_S: float = 60.0  # the nominal sequence takes about 11 s
POLL_S: float = 0.05


class CommandWatch:
    """Watches the command topic, to know which S_k is in force; publishes nothing."""

    def __init__(self) -> None:
        self._node = rclpy.create_node('run_v01')
        self._stages_seen: list[Stage | None] = []
        self._node.create_subscription(Command, COMMAND_TOPIC, self._on_command, LINK_QOS)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self) -> 'CommandWatch':
        self._spinning.start()
        return self

    def __exit__(self, *_exception: object) -> None:
        self._executor.shutdown()
        self._node.destroy_node()

    def has_begun(self) -> bool:
        return Stage.S1 in self._stages_seen

    def has_stopped(self) -> bool:
        """Tell whether S4 came after S1; the idle TWIST(0, 0) before the mission is no S4."""
        if not self.has_begun():
            return False
        return Stage.S4 in self._stages_seen[self._stages_seen.index(Stage.S1) :]

    def _on_command(self, command: Command) -> None:
        self._stages_seen.append(get_stage(command))


class V01Procedure:
    """The two executions of V-01, announced on the terminal."""

    def __init__(self, run_directory: Path):
        self._run_directory = run_directory

    def run(self) -> None:
        announce('V-01, nominal execution: keep the wheels free and the front of the sensor clear')
        self._run_execution(NOMINAL, on_begin=lambda: None, on_stop=lambda: None)
        announce('V-01, execution with the fault: lock one wheel by hand at LOCK NOW, during S1')
        self._run_execution(FAILURE, on_begin=announce_lock, on_stop=announce_release)

    def _run_execution(
        self, name: str, on_begin: Callable[[], None], on_stop: Callable[[], None]
    ) -> None:
        directory = self._run_directory / name
        directory.mkdir(parents=True)
        with BagRecorder(str(directory / BAG), RECORDED_TOPICS), CommandWatch() as watch:
            time.sleep(RECORDER_UP_S)
            with SequencerProcess(MISSION, directory / 'sequencer.log'):
                save_node_list_with_sequencer(directory / NODES_BEFORE)
                if wait_until(watch.has_begun, EXECUTION_TIMEOUT_S):
                    on_begin()
                if wait_until(watch.has_stopped, EXECUTION_TIMEOUT_S):
                    announce('S4: TWIST(0, 0)')
                    on_stop()
                else:
                    announce('no S4 before the timeout; check_v01 will judge the bag')
                time.sleep(T_OBS_S)
                save_node_list(directory / NODES_AFTER)


def save_node_list_with_sequencer(path: Path) -> None:
    """Save `ros2 node list` once the sequencer appears in it, or after the timeout."""
    deadline = time.monotonic() + NODE_DISCOVERY_TIMEOUT_S
    while SEQUENCER_NODE not in save_node_list(path) and time.monotonic() < deadline:
        time.sleep(POLL_S)


def save_node_list(path: Path) -> list[str]:
    nodes = subprocess.run(['ros2', 'node', 'list'], capture_output=True, text=True).stdout
    path.write_text(nodes)
    return nodes.split()


def wait_until(condition: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(POLL_S)
    return True


def announce_lock() -> None:
    announce('S1 began: LOCK NOW one wheel by hand and hold it')


def announce_release() -> None:
    announce('RELEASE the wheel')


def announce(text: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: run_v01 <run directory>')
        sys.exit(2)
    run_directory = Path(sys.argv[1])
    rclpy.init()
    try:
        with MicroRosAgent():
            announce('V-01: agent up; wheels suspended, deliberator not running?')
            time.sleep(SESSION_UP_S)
            V01Procedure(run_directory).run()
    finally:
        rclpy.try_shutdown()  # SIGINT: the handler of rclpy may have shut down already
    announce(f'done: check_v01 {run_directory}')


if __name__ == '__main__':
    main()
