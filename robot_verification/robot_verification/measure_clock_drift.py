"""
Bench measurement of the drift between the clocks of the ESP32 and the RPi (ADR 027).

Criticality: T1, evidence (robot-docs codigo.md).

What: the drift of the clock of the ESP32 against the clock of the agent, in ppm, and the error of
each clock synchronization, from which the resynchronization period is chosen.
How: the controller stamps each pose with the nominal release t_k of its cycle and the link
publishes the latest at 20 Hz, adding the offset of the last synchronization; between
synchronizations two stamps are therefore an exact multiple of T_c apart. The residual of each
interval to the nearest multiple of T_c is zero, but for a few us of conversion noise, except at a
synchronization, where it is the jump of the offset: the drift accumulated in the period plus the
difference of the errors of two synchronizations. The sum of the residuals is the change of the
reconstructed offset (stamp minus local time), so the drift is that sum over the local time. It
is taken between the first and the last jump, because the offset at either end of the recording
comes from a synchronization outside it; the jumps in between need not be told from the noise.
The jumps, above JUMP_THRESHOLD_NS, also give the spread.
Why: ADR 027, note (robot-docs adr/027), on the bag of V-07, with no change to the firmware.

Validity of the run: at least two jumps; every residual is below T_c/4, so the rounding to a
multiple of T_c is unambiguous; the number of jumps matches the duration over the synchronization
period, within one.

Usage: measure_clock_drift <bag directory>. Exit code: 0 measured, 2 invalid run.
"""

from __future__ import annotations

import itertools
import statistics
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import ControllerRecording, Verdict

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry

T_C_NS: int = 10_000_000  # period of the control task: the stamps are multiples of it
CLOCK_SYNC_PERIOD_S: float = 10.0  # CLOCK_SYNC_PERIOD of the link (robot-firmware config.h)
# Provisional, to be set from the histogram of the residuals of the V-07 bag (F7 I7): above the
# conversion noise (a few us, to be measured), below the jump of a drift of 2 ppm in 10 s (20 us).
JUMP_THRESHOLD_NS: int = 10_000
MAX_RESIDUAL_NS: int = T_C_NS // 4  # beyond this the nearest multiple of T_c is ambiguous
JUMP_COUNT_TOLERANCE: int = 1  # a synchronization at each end of the recording may be cut off
NS_PER_S: int = 1_000_000_000
NS_PER_US: int = 1000
PPM: float = 1e6


def get_stamp_ns(odometry: Odometry) -> int:
    stamp = odometry.header.stamp
    return stamp.sec * NS_PER_S + stamp.nanosec


def get_residual_ns(interval_ns: int) -> int:
    """Return the interval minus the nearest multiple of T_c."""
    return interval_ns - round(interval_ns / T_C_NS) * T_C_NS


class ClockDriftRun:
    """The residuals of the intervals between the Odometry stamps of a recording."""

    def __init__(self, recording: ControllerRecording):
        self.stamps_ns: list[int] = [get_stamp_ns(s.message) for s in recording.odometry]
        intervals_ns = [after - before for before, after in itertools.pairwise(self.stamps_ns)]
        self.residuals_ns: list[int] = [get_residual_ns(i) for i in intervals_ns]

    def has_samples(self) -> bool:
        return len(self.stamps_ns) >= 2

    def measure_drift_ppm(self) -> float:
        """
        Return the drift in ppm, positive if the clock of the ESP32 is slow.

        It is the change of the offset from just after the first jump to just after the last,
        over the local time between them: the offset at either end of the recording comes from a
        synchronization outside it.
        """
        first, last = self.find_first_and_last_jump()
        offset_change_ns = sum(self.residuals_ns[first + 1 : last + 1])
        stamps_apart_ns = self.stamps_ns[last + 1] - self.stamps_ns[first + 1]
        return offset_change_ns / (stamps_apart_ns - offset_change_ns) * PPM

    def find_first_and_last_jump(self) -> tuple[int, int]:
        """Return the positions in the residuals of the first and the last jump."""
        positions = [i for i, r in enumerate(self.residuals_ns) if abs(r) > JUMP_THRESHOLD_NS]
        return positions[0], positions[-1]

    def measure_duration_s(self) -> float:
        return (self.stamps_ns[-1] - self.stamps_ns[0]) / NS_PER_S

    def get_jumps_ns(self) -> list[int]:
        return [r for r in self.residuals_ns if abs(r) > JUMP_THRESHOLD_NS]

    def measure_noise_ns(self) -> int:
        """Return the largest residual that is not a jump: the conversion noise."""
        noise = [abs(r) for r in self.residuals_ns if abs(r) <= JUMP_THRESHOLD_NS]
        return max(noise, default=0)

    def measure_largest_residual_ns(self) -> int:
        return max((abs(r) for r in self.residuals_ns), default=0)


class ClockDriftMeasurement:
    """The validity conditions of the measurement, and its verdict."""

    def __init__(self, run: ClockDriftRun):
        self.run: ClockDriftRun = run

    @property
    def verdict(self) -> Verdict:
        return Verdict.PASS if self.run_is_valid else Verdict.INVALID

    @property
    def run_is_valid(self) -> bool:
        return (
            self.run.has_samples()
            and self.has_two_jumps()
            and self.is_rounding_unambiguous()
            and self.does_jump_count_match_the_period()
        )

    def has_two_jumps(self) -> bool:
        """Tell whether the drift can be taken between a first and a last jump."""
        return len(self.run.get_jumps_ns()) >= 2

    def is_rounding_unambiguous(self) -> bool:
        return self.run.measure_largest_residual_ns() < MAX_RESIDUAL_NS

    def does_jump_count_match_the_period(self) -> bool:
        expected = self.get_expected_jump_count()
        return abs(len(self.run.get_jumps_ns()) - expected) <= JUMP_COUNT_TOLERANCE

    def get_expected_jump_count(self) -> int:
        return int(self.run.measure_duration_s() // CLOCK_SYNC_PERIOD_S)


class ClockDriftReport:
    """The report: the drift, the jumps and the noise, and the validity of each check."""

    def __init__(self, measurement: ClockDriftMeasurement):
        self.measurement: ClockDriftMeasurement = measurement
        self.run: ClockDriftRun = measurement.run

    def __str__(self) -> str:
        lines = [f'clock drift: {len(self.run.stamps_ns)} Odometry stamps']
        if self.run.has_samples():
            lines += self._format_validity_lines() + self._format_measurement_lines()
        lines.append(f'clock drift: {self.measurement.verdict.name}')
        return '\n'.join(lines)

    def _format_validity_lines(self) -> list[str]:
        m = self.measurement
        largest_us = self.run.measure_largest_residual_ns() / NS_PER_US
        return [
            f'  duration {self.run.measure_duration_s():.1f} s',
            f'  largest residual {largest_us:.1f} us (max {MAX_RESIDUAL_NS / NS_PER_US:.0f} us), '
            f'passed: {m.is_rounding_unambiguous()}',
            f'  jumps {len(self.run.get_jumps_ns())} (expected {m.get_expected_jump_count()} '
            f'+- {JUMP_COUNT_TOLERANCE}), passed: {m.does_jump_count_match_the_period()}',
        ]

    def _format_measurement_lines(self) -> list[str]:
        if not self.measurement.has_two_jumps():
            return [f'  fewer than two jumps above {JUMP_THRESHOLD_NS / NS_PER_US:.0f} us']
        drift_ppm = self.run.measure_drift_ppm()
        lines = [
            f'  drift {drift_ppm:+.2f} ppm: {drift_ppm * CLOCK_SYNC_PERIOD_S:+.1f} us per '
            f'{CLOCK_SYNC_PERIOD_S:.0f} s period',
            f'  conversion noise up to {self.run.measure_noise_ns() / NS_PER_US:.1f} us '
            f'(jump threshold {JUMP_THRESHOLD_NS / NS_PER_US:.0f} us)',
        ]
        jumps_us = [j / NS_PER_US for j in self.run.get_jumps_ns()]
        if len(jumps_us) >= 2:
            lines.append(
                f'  jumps mean {statistics.mean(jumps_us):+.1f} us, spread (stdev) '
                f'{statistics.stdev(jumps_us):.1f} us, min {min(jumps_us):+.1f} us, '
                f'max {max(jumps_us):+.1f} us'
            )
        return lines


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: measure_clock_drift <bag directory>')
        sys.exit(Verdict.INVALID.value)
    measurement = ClockDriftMeasurement(ClockDriftRun(ControllerRecording.from_bag(sys.argv[1])))
    print(ClockDriftReport(measurement))
    sys.exit(measurement.verdict.value)


if __name__ == '__main__':
    main()
