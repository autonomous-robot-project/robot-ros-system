"""
V-07 procedure on the bench: conducts the trial and records the ros2 bag that check_v07 judges.

Criticality: T2, product (robot-docs codigo.md).

Steps of V-07: start the agent and the sequencer with the mission of V-07, which commands
TWIST(v_r, 0) and answers each activation of the test code with TWIST(0, 0); record the command,
the State, the Instrumentation and the Odometry; ground the test pin of the ESP32 (GPIO 12,
ADR 049), so that the controller generates the test events; wait for n events; stop.

The same bag serves check_v05, so the recording lasts at least T_e, and the clock drift of
ADR 027, which reads the stamps of the Odometry. The script only counts the events to know when to
stop: check_v07 judges them.

Usage, on the RPi with the wheels suspended and the motor switch on: run_v07 <bag directory>.
"""

import os

# Fast DDS only on UDP (ADR 045), set before ROS starts, so the children inherit it.
os.environ['FASTDDS_BUILTIN_TRANSPORTS'] = 'UDPv4'

import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import rclpy  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from robot_interfaces.msg import Instrumentation  # noqa: E402

from robot_verification.check_v05 import T_E_S  # noqa: E402
from robot_verification.check_v07 import N_EVENTS, V_R  # noqa: E402
from robot_verification.link import (  # noqa: E402
    LINK_QOS,
    BagRecorder,
    MicroRosAgent,
    SequencerProcess,
)
from robot_verification.recording import (  # noqa: E402
    COMMAND_TOPIC,
    INSTRUMENTATION_TOPIC,
    ODOMETRY_TOPIC,
    STATE_TOPIC,
)

RECORDED_TOPICS: list[str] = [COMMAND_TOPIC, STATE_TOPIC, INSTRUMENTATION_TOPIC, ODOMETRY_TOPIC]
MISSION: str = 'v07'
# Beside the bag, not inside: ros2 bag creates the directory itself and refuses an existing one.
SEQUENCER_LOG_SUFFIX: str = '_sequencer.log'
SESSION_UP_S: float = 5.0  # agent and recorder up, session of the ESP32 established
# check_v07 leaves out an event answered before the recording; the margin covers it.
EXTRA_EVENTS: int = 2
# 300 events of about 1.05 s take 5.3 min; beyond this the run is stopped and judged invalid.
MAX_TRIAL_S: float = 15 * 60.0
POLL_S: float = 1.0


class EventCounter:
    """Counts the test events answered since it started, from the index in the Instrumentation."""

    def __init__(self) -> None:
        self._node = rclpy.create_node('run_v07')
        self._node.create_subscription(
            Instrumentation, INSTRUMENTATION_TOPIC, self._on_instrumentation, LINK_QOS
        )
        self._first_index: int | None = None
        self._last_index: int = 0
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)

    def __enter__(self) -> 'EventCounter':
        self._spinning.start()
        return self

    def __exit__(self, *_exception: object) -> None:
        self._executor.shutdown()
        self._node.destroy_node()

    @property
    def count(self) -> int:
        if self._first_index is None:
            return 0
        return self._last_index - self._first_index

    def _on_instrumentation(self, message: Instrumentation) -> None:
        if self._first_index is None:
            self._first_index = message.test_event_index
        self._last_index = message.test_event_index


class V07Procedure:
    """The steps of V-07, announced on the terminal."""

    def __init__(self, counter: EventCounter):
        self._counter = counter

    def run(self) -> None:
        announce(f'sequencer at TWIST({V_R}, 0): ground GPIO 12 of the ESP32 now (jumper to GND)')
        started_s = time.monotonic()
        while not self._is_done(time.monotonic() - started_s):
            time.sleep(POLL_S)
        announce(f'{self._counter.count} events: release GPIO 12 now')

    def _is_done(self, elapsed_s: float) -> bool:
        has_events = self._counter.count >= N_EVENTS + EXTRA_EVENTS
        if elapsed_s >= MAX_TRIAL_S:
            announce(f'stopped after {MAX_TRIAL_S:.0f} s with {self._counter.count} events')
            return True
        return has_events and elapsed_s >= T_E_S


def get_sequencer_log_path(bag_path: str) -> str:
    return os.path.normpath(bag_path) + SEQUENCER_LOG_SUFFIX


def announce(text: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {text}', flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: run_v07 <bag directory>')
        sys.exit(2)
    rclpy.init()
    try:
        with (
            MicroRosAgent(),
            BagRecorder(sys.argv[1], RECORDED_TOPICS),
            SequencerProcess(MISSION, get_sequencer_log_path(sys.argv[1])),
            EventCounter() as counter,
        ):
            announce('V-07: agent, recorder and sequencer up; wheels suspended, motor switch on?')
            time.sleep(SESSION_UP_S)
            V07Procedure(counter).run()
    finally:
        # The SIGINT handler of rclpy may have shut the context down already.
        rclpy.try_shutdown()
    announce(f'done: check_v07 {sys.argv[1]} and check_v05 {sys.argv[1]}')


if __name__ == '__main__':
    main()
