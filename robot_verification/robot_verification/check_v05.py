"""
V-05: deadlines of the control task (RNF-01), evaluated on the ros2 bag recorded for V-07.

Criticality: T1, evidence (robot-docs codigo.md).

What: no control cycle ended after t_k + T_c during the recording; reports C_max, R_max, U and the
margin. How: the Instrumentation accumulates since boot (ADR 028), so the misses of the recording
are the difference between its last and first samples, and C_max and R_max, also since boot, are an
upper bound of the values of the recording. Why: V-05 has no run of its own; the recording of V-07
already carries the whole system and its load.

Usage: check_v05 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

import itertools
import sys
from typing import TYPE_CHECKING

from robot_verification.recording import ControllerRecording, Sample, Verdict

if TYPE_CHECKING:
    from robot_interfaces.msg import Instrumentation

# Parameters of V-05.
T_C_US: int = 10_000  # period and deadline of the control task
T_E_S: float = 300.0  # minimum duration of the recording: 30 000 cycles (Wilks, 1941)

# Conditions of the run.
# A restart of the ESP32 zeroes the accumulated values, and the restart with the reconnection of
# the session takes longer than this; a shorter gap is only lost messages, which the accumulated
# values tolerate (ADR 028).
MAX_INSTRUMENTATION_GAP_S: float = 1.0
# The meter is validated by reading the code and by its host test, not by an external instrument.
# Above half the period the margin is no longer large against the error of that validation, and
# the run needs the logic analyzer to conclude.
METER_VALIDATED_UP_TO_US: int = T_C_US // 2


def measure_gap_s(before: Sample[Instrumentation], after: Sample[Instrumentation]) -> float:
    return after.time_s - before.time_s


class DeadlineRun:
    """The quantities measured on the Instrumentation of a recording."""

    def __init__(self, recording: ControllerRecording):
        self.samples: list[Sample[Instrumentation]] = list(recording.instrumentation)

    def has_samples(self) -> bool:
        """Tell whether there are the two samples that bound the recording."""
        return len(self.samples) >= 2

    def measure_deadline_misses(self) -> int:
        """Return the misses during the recording: last minus first, as they count since boot."""
        return self.last.deadline_misses - self.first.deadline_misses

    def get_max_execution_time_us(self) -> int:
        """Return C_max since boot, an upper bound of the C_max of the recording."""
        return self.last.max_execution_time_us

    def get_max_response_time_us(self) -> int:
        """Return R_max since boot, an upper bound of the R_max of the recording."""
        return self.last.max_response_time_us

    def measure_duration_s(self) -> float:
        return self.samples[-1].time_s - self.samples[0].time_s

    def measure_longest_gap_s(self) -> float:
        pairs = itertools.pairwise(self.samples)
        return max(measure_gap_s(before, after) for before, after in pairs)

    @property
    def first(self) -> Instrumentation:
        return self.samples[0].message

    @property
    def last(self) -> Instrumentation:
        return self.samples[-1].message


class V05:
    """
    The criterion and the validity conditions of V-05, and its verdict.

    Valid run: the Instrumentation was recorded for at least T_e, with no gap that a restart of the
    ESP32 would leave, and R_max is within the range in which the meter is validated.
    """

    def __init__(self, run: DeadlineRun):
        self.run: DeadlineRun = run

    # --- Verdict ---

    @property
    def verdict(self) -> Verdict:
        if not self.run_is_valid:
            return Verdict.INVALID
        return Verdict.PASS if self.did_meet_every_deadline() else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        return (
            self.run.has_samples()
            and self.did_last_at_least_t_e()
            and self.was_recorded_without_restart()
            and self.is_response_time_within_meter_validation()
        )

    # --- Criterion ---

    def did_meet_every_deadline(self) -> bool:
        return self.run.measure_deadline_misses() == 0

    # --- Validity of the run ---

    def did_last_at_least_t_e(self) -> bool:
        return self.run.measure_duration_s() >= T_E_S

    def was_recorded_without_restart(self) -> bool:
        return self.run.measure_longest_gap_s() <= MAX_INSTRUMENTATION_GAP_S

    def is_response_time_within_meter_validation(self) -> bool:
        return self.run.get_max_response_time_us() <= METER_VALIDATED_UP_TO_US


class V05Report:
    """The report of V-05: the measured quantities and the verdict of each check."""

    def __init__(self, verification: V05):
        self.verification: V05 = verification
        self.run: DeadlineRun = verification.run

    def __str__(self) -> str:
        lines = [f'V-05: {len(self.run.samples)} Instrumentation samples']
        if self.run.has_samples():
            lines += self._format_validity_lines() + self._format_measurement_lines()
        lines.append(f'V-05: {self.verification.verdict.name}')
        return '\n'.join(lines)

    def _format_validity_lines(self) -> list[str]:
        return [
            f'  duration {self.run.measure_duration_s():.1f} s (min {T_E_S:.0f} s), '
            f'passed: {self.verification.did_last_at_least_t_e()}',
            f'  longest Instrumentation gap {self.run.measure_longest_gap_s():.3f} s '
            f'(max {MAX_INSTRUMENTATION_GAP_S:.1f} s), '
            f'passed: {self.verification.was_recorded_without_restart()}',
        ]

    def _format_measurement_lines(self) -> list[str]:
        c_max_us = self.run.get_max_execution_time_us()
        r_max_us = self.run.get_max_response_time_us()
        return [
            f'  deadline misses in the recording: {self.run.measure_deadline_misses()}',
            f'  C_max {c_max_us} us, U = C_max / T_c = {c_max_us / T_C_US:.1%} (since boot)',
            f'  R_max {r_max_us} us, margin (T_c - R_max) / T_c = '
            f'{(T_C_US - r_max_us) / T_C_US:.1%} (since boot; meter validated up to '
            f'{METER_VALIDATED_UP_TO_US} us), '
            f'passed: {self.verification.is_response_time_within_meter_validation()}',
        ]


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v05 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    run = DeadlineRun(ControllerRecording.from_bag(sys.argv[1]))
    verification = V05(run)
    print(V05Report(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
