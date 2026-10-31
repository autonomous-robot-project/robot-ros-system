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
NOISE_NS = 2_000  # conversion noise, uniform in +- this
SYNC_ERROR_NS = 30_000  # error of each synchronization, uniform in +- this


def odometry_stamped(stamp_ns):
    stamp = SimpleNamespace(sec=stamp_ns // NS_PER_S, nanosec=stamp_ns % NS_PER_S)
    return SimpleNamespace(header=SimpleNamespace(stamp=stamp))


@dataclass
class SyntheticDriftRun:
    """
    Odometry stamps of a link that drifts as configured.

    The local clock advances by the publication period, sometimes one T_c more or less; the offset
    is resynchronized every CLOCK_SYNC_PERIOD_S to the true one, with an error.
    """

    drift_ppm: float = DRIFT_PPM
    sync_error_ns: int = SYNC_ERROR_NS
    noise_ns: int = NOISE_NS
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
            noise_ns = rng.randint(-self.noise_ns, self.noise_ns)
            jitter_ns = self.jitter_ns if len(samples) == 100 else 0
            stamp_ns = local_ns + offset_ns + noise_ns + jitter_ns
            samples.append(Sample(local_ns / NS_PER_S, odometry_stamped(stamp_ns)))
            local_ns += self._next_interval_ns(rng)
        return ControllerRecording([], [], [], samples)

    def _next_interval_ns(self, rng: random.Random) -> int:
        if not self.uneven_publication:
            return PUBLISH_PERIOD_NS
        return PUBLISH_PERIOD_NS + rng.choice([-T_C_NS, 0, 0, 0, T_C_NS])


def measure(run: SyntheticDriftRun) -> ClockDriftMeasurement:
    return ClockDriftMeasurement(ClockDriftRun(run.recording()))


# The sync errors of the two ends, over the duration: 2 * 30 us / 360 s = 0.17 ppm.
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


def test_noise_is_reported_apart_from_the_jumps():
    run = measure(SyntheticDriftRun()).run
    assert run.measure_noise_ns() <= 2 * NOISE_NS


def test_residual_beyond_a_quarter_of_t_c_is_invalid():
    measurement = measure(SyntheticDriftRun(jitter_ns=T_C_NS // 4 + 1))
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
