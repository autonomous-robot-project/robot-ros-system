"""Synthetic messages and streams shared by the unit tests of the verification scripts."""

from types import SimpleNamespace

from robot_verification.recording import Sample

TWIST = 0  # Command.TWIST


def command(linear=0.2, angular=0.0):
    return SimpleNamespace(behavior=TWIST, linear=linear, angular=angular, theta=0.0, x=0.0, y=0.0)


def state(echo, faults=0):
    return SimpleNamespace(faults=faults, reflexes=0, echo=echo)


def instrumentation(expiries=0, expiry_delay_s=0.0, left_wheel_speed=0.0):
    return SimpleNamespace(
        expiries=expiries,
        expiry_delay_us=round(expiry_delay_s * 1e6),
        left_wheel_speed=left_wheel_speed,
        right_wheel_speed=0.0,
    )


def periodic(start_s, end_s, period_s, message_at):
    """Return samples every period_s from start_s to end_s, the message built for each time."""
    count = round((end_s - start_s) / period_s)
    times = [start_s + k * period_s for k in range(count)]
    return [Sample(t, message_at(t)) for t in times]
