"""Unit tests of the clock drift measurement on synthetic Odometry stamps."""

from dataclasses import dataclass, field
import random
from types import SimpleNamespace

import pytest
from robot_verification.measure_clock_drift import (
    CLOCK_SYNC_PERIOD_S,
    NS_PER_S,
    T_C_NS,
    ClockDriftMeasurement,
    ClockDriftReport,
    ClockDriftRun,
)
from robot_verification.recording import ControllerRecording, Sample, Verdict

PUBLISH_PERIOD_NS = 5 * T_C_NS  # 20 Hz
DURATION_S = 360.0  # the V-07 recording, about 6 min
DRIFT_PPM = 20.0
NOISE_NS = 5_000  # delay of a read with nothing in between, uniform in 0 to this
DELAYED_FRACTION = 0.2  # reads with a preemption or an interrupt in between
MAX_DELAY_NS = 700_000  # their delay, uniform in 0 to this (645 us seen, robot-docs nota 0018)
SYNC_ERROR_NS = 100_000  # error of each synchronization, uniform in +- this (nota 0018)


def odometry_stamped(stamp_ns):
    stamp = SimpleNamespace(sec=stamp_ns // NS_PER_S, nanosec=stamp_ns % NS_PER_S)
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp))


@dataclass
class SyntheticDriftRun:
    """
    Odometry stamps of a link that drifts as configured.

    The local clock advances by the publication period, sometimes one T_c more or less; the offset
    is resynchronized every CLOCK_SYNC_PERIOD_S to the true one, with an error. Each stamp is
    delayed by its read, never advanced.
    """

    drift_ppm: float = DRIFT_PPM
    sync_error_ns: int = SYNC_ERROR_NS
    noise_ns: int = NOISE_NS
    delayed_fraction: float = DELAYED_FRACTION
    uneven_publication: bool = False  # 40 and 60 ms intervals now and then
    jitter_ns: int = 0  # extra jitter of a stamp, beyond the rounding range when large
    seed: int = 1
    stamps_ns: list[int] = field(init=False)

    def recording(self) -> ControllerRecording:
        rng = random.Random(self.seed)
        samples = []
        local_ns, offset_ns, next_sync_ns = 0, 0, 0
        while local_ns < DURATION_S * NS_PER_S:
            if local_ns >= next_sync_ns:
                true_offset_ns = round(local_ns * self.drift_ppm * 1e-6)
                offset_ns = true_offset_ns + rng.randint(-self.sync_error_ns, self.sync_error_ns)
                next_sync_ns += round(CLOCK_SYNC_PERIOD_S * NS_PER_S)
            jitter_ns = self.jitter_ns if len(samples) == 100 else 0
            stamp_ns = local_ns + offset_ns + self._read_delay_ns(rng) + jitter_ns
            samples.append(Sample(local_ns / NS_PER_S, odometry_stamped(stamp_ns)))
            local_ns += self._next_interval_ns(rng)
        return ControllerRecording([], [], [], samples)

    def _read_delay_ns(self, rng: random.Random) -> int:
        if rng.random() < self.delayed_fraction:
            return rng.randint(0, MAX_DELAY_NS)
        return rng.randint(0, self.noise_ns)

    def _next_interval_ns(self, rng: random.Random) -> int:
        if not self.uneven_publication:
            return PUBLISH_PERIOD_NS
        return PUBLISH_PERIOD_NS + rng.choice([-T_C_NS, 0, 0, 0, T_C_NS])


def measure(run: SyntheticDriftRun) -> ClockDriftMeasurement:
    return ClockDriftMeasurement(ClockDriftRun(run.recording()))


# The sync errors of the two ends, over the duration: 2 * 100 us / 360 s = 0.56 ppm.
DRIFT_TOLERANCE_PPM = 2 * SYNC_ERROR_NS / (DURATION_S * NS_PER_S) * 1e6


@pytest.mark.parametrize('drift_ppm', [DRIFT_PPM, -DRIFT_PPM])
def test_drift_is_recovered(drift_ppm):
    measurement = measure(SyntheticDriftRun(drift_ppm=drift_ppm))
    assert measurement.verdict is Verdict.PASS
    assert measurement.run.measure_drift_ppm() == pytest.approx(drift_ppm, abs=DRIFT_TOLERANCE_PPM)


def test_intervals_of_other_multiples_of_t_c_do_not_bias_the_drift():
    measurement = measure(SyntheticDriftRun(uneven_publication=True))
    assert measurement.verdict is Verdict.PASS
    assert measurement.run.measure_drift_ppm() == pytest.approx(DRIFT_PPM, abs=DRIFT_TOLERANCE_PPM)


def test_one_jump_per_synchronization():
    run = measure(SyntheticDriftRun()).run
    assert len(run.get_jumps_ns()) == pytest.approx(DURATION_S / CLOCK_SYNC_PERIOD_S, abs=1)


def test_window_lows_stay_on_their_plateau_despite_delayed_reads():
    run = measure(SyntheticDriftRun()).run
    assert run.measure_noise_ns() <= NOISE_NS


def test_delayed_reads_alone_are_no_jumps():
    # Every read delayed up to 700 us, but no drift and exact synchronizations.
    run = measure(SyntheticDriftRun(drift_ppm=0.0, sync_error_ns=0)).run
    assert run.get_jumps_ns() == []


def test_residual_beyond_a_quarter_of_t_c_is_invalid():
    # Beyond T_c/4 even when the read before it was delayed the most.
    measurement = measure(SyntheticDriftRun(jitter_ns=T_C_NS // 4 + MAX_DELAY_NS + 1))
    assert measurement.verdict is Verdict.INVALID


def test_jumps_hidden_under_the_threshold_are_invalid():
    # No drift and exact synchronizations: no jump is seen, so the count does not match.
    measurement = measure(SyntheticDriftRun(drift_ppm=0.0, sync_error_ns=0))
    assert measurement.verdict is Verdict.INVALID


def test_report_shows_the_drift_and_the_jumps():
    report = str(ClockDriftReport(measure(SyntheticDriftRun())))
    assert 'drift +20.' in report or 'drift +19.' in report
    assert 'jumps mean' in report
    assert report.endswith('clock drift: PASS')
