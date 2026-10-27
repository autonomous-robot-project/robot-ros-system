"""
Shared pieces of the verification scripts.

Criticality: T1, evidence (robot-docs codigo.md).

The verdict, the samples of one recorded topic, the reading of a ros2 bag and the recording of the
controller topics. ROS is imported only to read a bag, so the criteria can be tested without ROS;
the message types below are for the type checker only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robot_interfaces.msg import Command, Instrumentation, State

COMMAND_TOPIC = '/controller/command'
STATE_TOPIC = '/controller/state'
INSTRUMENTATION_TOPIC = '/controller/instrumentation'

# The float32 fields of the messages keep ~7 significant digits: 0.2 arrives as 0.20000000298.
FLOAT32_RELATIVE_TOLERANCE = 1e-6


def is_close_float32(received: float, expected: float) -> bool:
    """Tell whether a float32 field of a message carries the expected value."""
    return math.isclose(received, expected, rel_tol=FLOAT32_RELATIVE_TOLERANCE)


class Verdict(Enum):
    """Outcome of a run; the value is the exit code."""

    PASS = 0
    FAIL = 1
    INVALID = 2


@dataclass(frozen=True)
class Sample[MessageT]:
    """A recorded message and its reception time."""

    time_s: float
    message: MessageT


class Stream[MessageT]:
    """The samples of one topic, in reception order."""

    def __init__(self, samples: Iterable[Sample[MessageT]]):
        self.samples: list[Sample[MessageT]] = list(samples)

    def __iter__(self) -> Iterator[Sample[MessageT]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample[MessageT]:
        return self.samples[index]

    def at(self, time_s: float) -> Sample[MessageT] | None:
        """Return the latest sample received up to time_s, or None."""
        return next((s for s in reversed(self.samples) if s.time_s <= time_s), None)

    def between(self, start_s: float, end_s: float) -> list[Sample[MessageT]]:
        """Return the samples received after start_s and before end_s."""
        return [s for s in self.samples if start_s < s.time_s < end_s]


def read_bag(path: str, message_types: dict[str, type]) -> dict[str, Stream]:
    """Read a ros2 bag into one Stream per topic of message_types ({topic: message class})."""
    from rclpy.serialization import deserialize_message
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path), rosbag2_py.ConverterOptions('', ''))
    samples: dict[str, list[Sample]] = {topic: [] for topic in message_types}
    while reader.has_next():
        topic, data, time_ns = reader.read_next()
        if topic in message_types:
            message = deserialize_message(data, message_types[topic])
            samples[topic].append(Sample(time_ns * 1e-9, message))
    return {topic: Stream(topic_samples) for topic, topic_samples in samples.items()}


class ControllerRecording:
    """The commands, states and instrumentation of a run, in reception order."""

    def __init__(
        self,
        commands: Iterable[Sample[Command]],
        states: Iterable[Sample[State]],
        instrumentation: Iterable[Sample[Instrumentation]],
    ):
        self.commands: Stream[Command] = Stream(commands)
        self.states: Stream[State] = Stream(states)
        self.instrumentation: Stream[Instrumentation] = Stream(instrumentation)

    @classmethod
    def from_bag(cls, path: str) -> ControllerRecording:
        # ROS imports here, so the criteria can be tested without ROS.
        from robot_interfaces.msg import Command, Instrumentation, State

        streams = read_bag(
            path,
            {COMMAND_TOPIC: Command, STATE_TOPIC: State, INSTRUMENTATION_TOPIC: Instrumentation},
        )
        return cls(streams[COMMAND_TOPIC], streams[STATE_TOPIC], streams[INSTRUMENTATION_TOPIC])
