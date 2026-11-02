"""
Bench measurement of the drift between the clocks of the ESP32 and the RPi (ADR 027).

Criticality: T1, evidence (robot-docs codigo.md).

What: the drift of the clock of the ESP32 against the clock of the agent, in ppm, and the error of
each clock synchronization, from which the resynchronization period is chosen.
How: the controller stamps each pose with the nominal release t_k of its cycle, and the link
publishes the latest at 20 Hz, carried to the clock of the agent with the offset of the last
synchronization. The link reads the local clock before the clock of micro-ROS, so whatever runs
between the two reads (a preemption, an interrupt) makes the stamp later, never earlier. The phase
of a stamp, the sum of the residuals of the intervals to the nearest multiple of T_c, is therefore
the offset plus that delay. Its lowest value in each window of WINDOW_NS is flat between
synchronizations and steps at each one, by the drift accumulated in the period plus the difference
of the errors of two synchronizations. The offset of each plateau is the median of its window
lows; the drift is the change of the offset from the plateau after the first step to the plateau
after the last, over the time between them, because the offset before the first step comes from a
synchronization outside the recording. The steps also give the spread.
Why: ADR 027, note (robot-docs adr/027), on the bag of V-07, with no change to the firmware; the
delay of one side was found on bench bags (robot-docs nota 0018).

Validity of the run: at least two steps; every residual is below T_c/4, so the rounding to a
multiple of T_c is unambiguous (a restart of the ESP32 moves the phase by milliseconds); the number
of steps matches the duration over the synchronization period, within one.

Usage: measure_clock_drift <bag directory>. Exit code: 0 measured, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import statistics
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import ControllerRecording, Verdict

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry

T_C_NS: int = 10_000_000  # period of the control task: the stamps are multiples of it
CLOCK_SYNC_PERIOD_S: float = 10.0  # CLOCK_SYNC_PERIOD of the link (robot-firmware config.h)
# 20 stamps at 20 Hz: the lowest is one read without delay (robot-docs nota 0018).
WINDOW_NS: int = 1_000_000_000
# The window lows of a plateau stay within 16 us of each other, and the smallest step seen was
# 21 us (robot-docs nota 0018); a smaller step is lost and shows in the count of steps.
JUMP_THRESHOLD_NS: int = 20_000
MAX_RESIDUAL_NS: int = T_C_NS // 4  # beyond this the nearest multiple of T_c is ambiguous
JUMP_COUNT_TOLERANCE: int = 1  # a synchronization at each end of the recording may be cut off
NS_PER_S: int = 1_000_000_000
NS_PER_US: int = 1000
PPM: float = 1e6


@dataclass(frozen=True)
class Window:
    """The lowest phase of the stamps in one window, the window counted from the first stamp."""

    start_ns: int
    lowest_phase_ns: int


def get_stamp_ns(odometry: Odometry) -> int:
    stamp = odometry.header.stamp
    return stamp.sec * NS_PER_S + stamp.nanosec


def get_residual_ns(interval_ns: int) -> int:
    """Return the interval minus the nearest multiple of T_c."""
    return interval_ns - round(interval_ns / T_C_NS) * T_C_NS


def find_lowest_phase_per_window(stamps_ns: list[int], phases_ns: list[int]) -> list[Window]:
    """Return the lowest phase in each window of WINDOW_NS that has a stamp, in order."""
    lowest: dict[int, int] = {}
    for stamp_ns, phase_ns in zip(stamps_ns, phases_ns, strict=True):
        window = (stamp_ns - stamps_ns[0]) // WINDOW_NS
        lowest[window] = min(lowest.get(window, phase_ns), phase_ns)
    return [Window(window * WINDOW_NS, phase) for window, phase in sorted(lowest.items())]


class ClockDriftRun:
    """The phases of the Odometry stamps of a recording, and their lows per window."""

    def __init__(self, recording: ControllerRecording):
        self.stamps_ns: list[int] = [get_stamp_ns(s.message) for s in recording.odometry]
        intervals_ns = [after - before for before, after in itertools.pairwise(self.stamps_ns)]
        self.residuals_ns: list[int] = [get_residual_ns(i) for i in intervals_ns]
        # The phase of each stamp from the first: the offset plus the delay of its read.
        self.phases_ns: list[int] = list(itertools.accumulate(self.residuals_ns, initial=0))
        self.windows: list[Window] = (
            find_lowest_phase_per_window(self.stamps_ns, self.phases_ns) if self.stamps_ns else []
        )

    def has_samples(self) -> bool:
        return len(self.stamps_ns) >= 2

    def measure_drift_ppm(self) -> float:
        """
        Return the drift in ppm, positive if the clock of the ESP32 is slow.

        It is the change of the offset from the plateau after the first step to the plateau after
        the last, over the time between the two steps.
        """
        first, last = self.find_first_and_last_jump()
        offsets_ns = self.get_plateau_offsets_ns()
        offset_change_ns = offsets_ns[-1] - offsets_ns[1]
        windows_apart_ns = self.windows[last].start_ns - self.windows[first].start_ns
        return offset_change_ns / (windows_apart_ns - offset_change_ns) * PPM

    def find_first_and_last_jump(self) -> tuple[int, int]:
        """Return the positions in the windows of the first and the last step."""
        positions = self.find_jump_positions()
        return positions[0], positions[-1]

    def measure_duration_s(self) -> float:
        return (self.stamps_ns[-1] - self.stamps_ns[0]) / NS_PER_S

    def get_jumps_ns(self) -> list[int]:
        """Return each step: the offset of a plateau minus that of the plateau before it."""
        offsets_ns = self.get_plateau_offsets_ns()
        return [after - before for before, after in itertools.pairwise(offsets_ns)]

    def get_plateau_offsets_ns(self) -> list[int]:
        """Return the offset of each plateau between steps: the median of its window lows."""
        return [
            round(statistics.median(w.lowest_phase_ns for w in plateau))
            for plateau in self.split_into_plateaus()
        ]

    def measure_noise_ns(self) -> int:
        """Return how far a window low strays from the offset of its plateau."""
        offsets_ns = self.get_plateau_offsets_ns()
        return max(
            (
                abs(w.lowest_phase_ns - offset_ns)
                for plateau, offset_ns in zip(self.split_into_plateaus(), offsets_ns, strict=True)
                for w in plateau
            ),
            default=0,
        )

    def split_into_plateaus(self) -> list[list[Window]]:
        """Return the windows between consecutive steps; the first plateau ends at the first."""
        bounds = [0, *self.find_jump_positions(), len(self.windows)]
        return [self.windows[start:end] for start, end in itertools.pairwise(bounds)]

    def measure_largest_residual_ns(self) -> int:
        return max((abs(r) for r in self.residuals_ns), default=0)

    def find_jump_positions(self) -> list[int]:
        """Return the windows whose low differs from the one before by more than the threshold."""
        return [
            position
            for position, (before, after) in enumerate(itertools.pairwise(self.windows), start=1)
            if abs(after.lowest_phase_ns - before.lowest_phase_ns) > JUMP_THRESHOLD_NS
        ]


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
            f'  window lows within {self.run.measure_noise_ns() / NS_PER_US:.1f} us of their '
            f'plateau (jump threshold {JUMP_THRESHOLD_NS / NS_PER_US:.0f} us)',
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
