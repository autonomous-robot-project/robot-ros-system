"""Synthetic messages and streams shared by the unit tests of the verification scripts."""

import struct
from types import SimpleNamespace

from robot_verification.recording import Sample

TWIST = 0  # Command.TWIST


def to_float32(value):
    """Round to float32, as the float32 fields of the messages arrive from the bag."""
    return struct.unpack('f', struct.pack('f', value))[0]


def command(linear=0.2, angular=0.0):
    return SimpleNamespace(
        behavior=TWIST,
        linear=to_float32(linear),
        angular=to_float32(angular),
        theta=0.0,
        x=0.0,
        y=0.0,
    )


def state(echo, faults=0, reflexes=0):
    return SimpleNamespace(faults=faults, reflexes=reflexes, echo=echo)


def odometry(linear=0.0):
    """Return an Odometry with only the linear speed of its twist."""
    return SimpleNamespace(
        twist=SimpleNamespace(twist=SimpleNamespace(linear=SimpleNamespace(x=linear)))
    )


def instrumentation(
    expiries=0,
    expiry_delay_s=0.0,
    left_wheel_speed=0.0,
    right_wheel_speed=0.0,
    max_execution_time_us=0,
    max_response_time_us=0,
    deadline_misses=0,
):
    return SimpleNamespace(
        expiries=expiries,
        expiry_delay_us=round(expiry_delay_s * 1e6),
        max_execution_time_us=max_execution_time_us,
        max_response_time_us=max_response_time_us,
        deadline_misses=deadline_misses,
        left_wheel_speed=left_wheel_speed,
        right_wheel_speed=right_wheel_speed,
    )


def periodic(start_s, end_s, period_s, message_at):
    """Return samples every period_s from start_s to end_s, the message built for each time."""
    count = round((end_s - start_s) / period_s)
    times = [start_s + k * period_s for k in range(count)]
    return [Sample(t, message_at(t)) for t in times]
