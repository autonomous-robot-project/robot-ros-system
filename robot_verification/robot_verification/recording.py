"""
Shared pieces of the verification scripts.

The verdict, the samples of one recorded topic and the reading of a ros2 bag. ROS is imported only
to read a bag, so the criteria can be tested without ROS.
"""

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    """Outcome of a run; the value is the exit code."""

    PASS = 0
    FAIL = 1
    INVALID = 2


@dataclass(frozen=True)
class Sample:
    """A recorded message and its reception time."""

    time_s: float
    message: object


class Stream:
    """The samples of one topic, in reception order."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __iter__(self):
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def at(self, time_s):
        """Return the latest sample received up to time_s, or None."""
        return next((s for s in reversed(self.samples) if s.time_s <= time_s), None)

    def between(self, start_s, end_s):
        """Return the samples received after start_s and before end_s."""
        return [s for s in self.samples if start_s < s.time_s < end_s]


def read_bag(path, message_types):
    """Read a ros2 bag into one Stream per topic of message_types ({topic: message class})."""
    from rclpy.serialization import deserialize_message
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=path), rosbag2_py.ConverterOptions('', ''))
    samples = {topic: [] for topic in message_types}
    while reader.has_next():
        topic, data, time_ns = reader.read_next()
        if topic in message_types:
            message = deserialize_message(data, message_types[topic])
            samples[topic].append(Sample(time_ns * 1e-9, message))
    return {topic: Stream(topic_samples) for topic, topic_samples in samples.items()}
