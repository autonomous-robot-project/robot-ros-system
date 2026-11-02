"""
V-08 procedure on the bench, one condition: records the bag of check_v05, check_v07, check_v08_c1.

Criticality: T2, product (robot-docs codigo.md).

Steps of V-08, for C1 or C2: establish the condition; record the command, the State, the
Odometry and the Instrumentation, and in C1 the plan; run V-07 for 10 min; stop.
- C1: the launch of the deliberator of C1 (SLAM on the LD19, planner over M), and the sequencer
  with the mission of V-08 asks for plans in a loop (F8 D2). The condition holds once the first
  plan arrives, before the recording starts, as check_v08_c1 expects.
- C2: stress-ng on the 4 cores of the RPi, with the active cooler.
The mission of V-08 is that of V-07 (TWIST(v_r, 0), TWIST(0, 0) at each test event), with the
heartbeat of the notebook off (V-08). Beside the bag, for the summary of #26 (F8 D5): pidstat of
every process every 5 s with its averages, free -m at the start and the end, and in C2 vcgencmd
get_throttled before and after, for information only.

Usage, on the RPi with the wheels suspended and the motor switch on:
run_v08 <C1|C2> <bag directory>.
"""

import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so the children inherit it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

from enum import Enum  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from nav_msgs.msg import Path  # noqa: E402
import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402

from robot_verification.check_v04 import PLAN_TOPIC  # noqa: E402
from robot_verification.link import (  # noqa: E402
    BagRecorder,
    MicroRosAgent,
    RosLaunch,
    SequencerProcess,
)
from robot_verification.recording import (  # noqa: E402
    COMMAND_TOPIC,
    INSTRUMENTATION_TOPIC,
    ODOMETRY_TOPIC,
    STATE_TOPIC,
)


class Condition(Enum):
    C1 = 'C1'  # deliberator computing
    C2 = 'C2'  # embedded computer overloaded


MISSION: str = 'v08'
CONTROLLER_TOPICS: list[str] = [COMMAND_TOPIC, STATE_TOPIC, INSTRUMENTATION_TOPIC, ODOMETRY_TOPIC]
C1_LAUNCH = ('robot_bringup', 'deliberator_c1.launch.py')
C2_LOAD: list[str] = ['stress-ng', '--cpu', '4']  # 100 % on the 4 cores (V-08)
PROCESS_STATS: list[str] = ['pidstat', '-u', '-r', '5']  # CPU and memory per process (F8 D5)
# Beside the bag, not inside: ros2 bag creates the directory itself and refuses an existing one.
SEQUENCER_LOG_SUFFIX: str = '_sequencer.log'
LOAD_LOG_SUFFIX: str = '_stress.log'
PROCESS_STATS_SUFFIX: str = '_pidstat.txt'
SYSTEM_SUFFIX: str = '_system.txt'
RUN_S: float = 10 * 60.0  # 60 000 control cycles and about 600 test events (V-08)
SESSION_UP_S: float = 5.0  # agent and sequencer up, session of the ESP32 established
# The launch of C1 starts the LD19, the SLAM and the planner; it has not been timed (nota 0019).
WAIT_FOR_PLAN_S: float = 60.0
PROGRESS_PERIOD_S: float = 60.0
PLAN_QUEUE_DEPTH = 1


class LoggedProcess:
    """A tool run as a child process, its output in a file, stopped by SIGINT."""

    def __init__(self, command: list[str], log_path: str):
        self._command = command
        self._log_path = log_path
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> 'LoggedProcess':
        with open(self._log_path, 'w') as log:
            self._process = subprocess.Popen(self._command, stdout=log, stderr=subprocess.STDOUT)
        return self

    def __exit__(self, *_exception: object) -> None:
        # SIGINT: stress-ng stops its workers, pidstat writes its averages.
        assert self._process is not None
        self._process.send_signal(signal.SIGINT)
        self._process.wait()


class PlanWatcher:
    """Tells whether the planner has delivered a plan since it started."""

    def __init__(self) -> None:
        self._node = rclpy.create_node('run_v08')
        self._node.create_subscription(Path, PLAN_TOPIC, self._on_plan, PLAN_QUEUE_DEPTH)
        self._has_plan = threading.Event()
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self) -> 'PlanWatcher':
        self._spinning.start()
        return self

    def __exit__(self, *_exception: object) -> None:
        self._executor.shutdown()
        self._node.destroy_node()

    def wait_for_plan(self, timeout_s: float) -> bool:
        return self._has_plan.wait(timeout_s)

    def _on_plan(self, _: Path) -> None:
        self._has_plan.set()


class V08Procedure:
    """The steps of V-08 for one condition, announced on the terminal."""

    def __init__(self, condition: Condition, bag_path: str):
        self._condition = condition
        self._bag_path = bag_path

    def run(self) -> None:
        write_system_state(self._bag_path, 'start', self._condition)
        with self._establish_condition():
            with (
                MicroRosAgent(),
                SequencerProcess(MISSION, get_beside_bag(self._bag_path, SEQUENCER_LOG_SUFFIX)),
            ):
                time.sleep(SESSION_UP_S)
                if self._condition is Condition.C1:
                    self._wait_for_plan()
                self._record()
        write_system_state(self._bag_path, 'end', self._condition)

    def _establish_condition(self) -> RosLaunch | LoggedProcess:
        if self._condition is Condition.C1:
            announce('C1: launching the deliberator of C1 (LD19, SLAM, planner over M)')
            return RosLaunch(*C1_LAUNCH)
        announce('C2: stress-ng on the 4 cores; is the cooler on?')
        return LoggedProcess(C2_LOAD, get_beside_bag(self._bag_path, LOAD_LOG_SUFFIX))

    def _wait_for_plan(self) -> None:
        with PlanWatcher() as watcher:
            if not watcher.wait_for_plan(WAIT_FOR_PLAN_S):
                raise RuntimeError(f'no plan in {WAIT_FOR_PLAN_S:.0f} s: C1 not established')
        announce('C1 established: the planner delivers plans')

    def _record(self) -> None:
        topics = CONTROLLER_TOPICS + ([PLAN_TOPIC] if self._condition is Condition.C1 else [])
        stats_path = get_beside_bag(self._bag_path, PROCESS_STATS_SUFFIX)
        with BagRecorder(self._bag_path, topics), LoggedProcess(PROCESS_STATS, stats_path):
            announce(f'recording {RUN_S / 60:.0f} min: ground GPIO 12 of the ESP32 now')
            wait_with_progress(RUN_S)
            announce('release GPIO 12 now')


def get_beside_bag(bag_path: str, suffix: str) -> str:
    return os.path.normpath(bag_path) + suffix


def write_system_state(bag_path: str, moment: str, condition: Condition) -> None:
    """Append free -m and, in C2, vcgencmd get_throttled to the system file beside the bag."""
    commands = [['free', '-m']]
    if condition is Condition.C2:
        commands.append(['vcgencmd', 'get_throttled'])
    with open(get_beside_bag(bag_path, SYSTEM_SUFFIX), 'a') as system:
        for command in commands:
            output = subprocess.run(command, capture_output=True, text=True).stdout
            system.write(f'--- {moment}: {" ".join(command)}\n{output}')


def wait_with_progress(duration_s: float) -> None:
    started_s = time.monotonic()
    while (left_s := duration_s - (time.monotonic() - started_s)) > 0:
        time.sleep(min(PROGRESS_PERIOD_S, left_s))
        announce(f'{(time.monotonic() - started_s) / 60:.0f} of {duration_s / 60:.0f} min')


def find_missing_tools(condition: Condition) -> list[str]:
    tools = [PROCESS_STATS[0], 'free'] + (
        [C2_LOAD[0], 'vcgencmd'] if condition is Condition.C2 else []
    )
    return [tool for tool in tools if shutil.which(tool) is None]


def announce(text: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def main() -> None:
    conditions = [condition.value for condition in Condition]
    if len(sys.argv) != 3 or sys.argv[1] not in conditions:
        print(f'usage: run_v08 <{"|".join(conditions)}> <bag directory>')
        sys.exit(2)
    condition, bag_path = Condition(sys.argv[1]), sys.argv[2]
    missing = find_missing_tools(condition)
    if missing:
        print(f'missing on this computer: {", ".join(missing)}')
        sys.exit(2)
    rclpy.init()
    try:
        V08Procedure(condition, bag_path).run()
    finally:
        rclpy.shutdown()
    checks = f'check_v05 {bag_path}, check_v07 {bag_path}'
    if condition is Condition.C1:
        checks += f', check_v08_c1 {bag_path}'
    announce(f'done: {checks}')


if __name__ == '__main__':
    main()
