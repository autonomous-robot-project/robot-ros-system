"""Unit tests of the V-05 criterion on synthetic Instrumentation streams."""

from dataclasses import dataclass

from robot_verification.check_v05 import (
    METER_VALIDATED_UP_TO_US,
    T_E_S,
    V05,
    DeadlineRun,
    V05Report,
)
from robot_verification.recording import ControllerRecording, Verdict
from synthetic import instrumentation, periodic

INSTRUMENTATION_PERIOD_S = 0.1  # 10 Hz
RESTART_S = 3.0  # no Instrumentation while the ESP32 restarts and the session comes back


@dataclass
class SyntheticV05Run:
    """
    The Instrumentation of a V-07 recording, from a controller that behaves as configured.

    The misses during the recording happen at its middle; those before it are already in the
    counter of the first sample, since the counter accumulates since boot.
    """

    duration_s: float = T_E_S + 10.0
    misses_before_recording: int = 0
    misses_during_recording: int = 0
    max_execution_time_us: int = 300
    max_response_time_us: int = 400
    gap_at_s: float | None = None  # start of a gap in the Instrumentation
    gap_s: float = RESTART_S

    def recording(self) -> ControllerRecording:
        period_s = INSTRUMENTATION_PERIOD_S
        samples = periodic(0.0, self.duration_s, period_s, self._instrumentation_at)
        kept = [s for s in samples if not self._is_in_gap(s.time_s)]
        return ControllerRecording([], [], kept)

    def _instrumentation_at(self, t: float):
        misses = self.misses_before_recording
        if t >= self.duration_s / 2:
            misses += self.misses_during_recording
        return instrumentation(
            max_execution_time_us=self.max_execution_time_us,
            max_response_time_us=self.max_response_time_us,
            deadline_misses=misses,
        )

    def _is_in_gap(self, t: float) -> bool:
        return self.gap_at_s is not None and self.gap_at_s <= t < self.gap_at_s + self.gap_s


def judge(run: SyntheticV05Run) -> V05:
    return V05(DeadlineRun(run.recording()))


def test_run_with_every_deadline_met_passes():
    assert judge(SyntheticV05Run()).verdict is Verdict.PASS


def test_a_late_cycle_in_the_recording_fails():
    assert judge(SyntheticV05Run(misses_during_recording=1)).verdict is Verdict.FAIL


def test_misses_before_the_recording_do_not_count():
    run = SyntheticV05Run(misses_before_recording=3)
    assert judge(run).verdict is Verdict.PASS


def test_recording_shorter_than_t_e_is_invalid():
    run = SyntheticV05Run(duration_s=T_E_S - 10.0)
    assert judge(run).verdict is Verdict.INVALID


def test_restart_of_the_esp32_is_invalid():
    run = SyntheticV05Run(gap_at_s=100.0)
    assert judge(run).verdict is Verdict.INVALID


def test_lost_messages_do_not_invalidate_the_run():
    run = SyntheticV05Run(gap_at_s=100.0, gap_s=0.5)
    assert judge(run).verdict is Verdict.PASS


def test_response_time_above_the_validated_range_is_invalid():
    run = SyntheticV05Run(max_response_time_us=METER_VALIDATED_UP_TO_US + 1)
    assert judge(run).verdict is Verdict.INVALID


def test_recording_without_instrumentation_is_invalid():
    empty = V05(DeadlineRun(ControllerRecording([], [], [])))
    assert empty.verdict is Verdict.INVALID
    assert 'INVALID' in str(V05Report(empty))


def test_report_shows_the_measurements_and_the_verdict():
    report = str(V05Report(judge(SyntheticV05Run(misses_during_recording=2))))
    assert 'deadline misses in the recording: 2' in report
    assert 'C_max 300 us' in report
    assert 'R_max 400 us' in report
    assert report.endswith('V-05: FAIL')
