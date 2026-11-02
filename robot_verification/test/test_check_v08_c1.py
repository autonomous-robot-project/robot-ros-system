"""Unit tests of the establishment of C1 (V-08) on synthetic plan and Instrumentation streams."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from robot_verification.check_v08_c1 import T_GAP_S, C1Establishment, C1Report, PlanDeliveryRun
from robot_verification.recording import Sample, Verdict
from synthetic import instrumentation, periodic

INSTRUMENTATION_PERIOD_S = 0.1  # 10 Hz
PLAN_PERIOD_S = 0.2  # one plan per response of the planner
RUN_S = 600.0  # 10 min of V-07


@dataclass
class SyntheticC1Run:
    """A C1 recording: the Instrumentation over the run and a plan every PLAN_PERIOD_S."""

    first_plan_s: float = PLAN_PERIOD_S
    last_plan_s: float = RUN_S - PLAN_PERIOD_S
    gap_at_s: float | None = None  # start of a gap in the plans
    gap_s: float = 0.0
    with_plans: bool = True

    def run(self) -> PlanDeliveryRun:
        reports = periodic(0.0, RUN_S, INSTRUMENTATION_PERIOD_S, lambda _t: instrumentation())
        return PlanDeliveryRun(self._plans(), reports)

    def _plans(self) -> list[Sample]:
        if not self.with_plans:
            return []
        plans = periodic(
            self.first_plan_s, self.last_plan_s, PLAN_PERIOD_S, lambda _t: SimpleNamespace()
        )
        return [p for p in plans if not self._is_in_gap(p.time_s)]

    def _is_in_gap(self, time_s: float) -> bool:
        return self.gap_at_s is not None and self.gap_at_s < time_s < self.gap_at_s + self.gap_s


def judge(**run) -> C1Establishment:
    return C1Establishment(SyntheticC1Run(**run).run())


def test_plans_through_the_run_establish_c1():
    assert judge().verdict == Verdict.PASS


@pytest.mark.parametrize(
    'run',
    [
        pytest.param({'gap_at_s': 300.0, 'gap_s': 2.0}, id='gap between plans'),
        pytest.param({'first_plan_s': 5.0}, id='first plan late'),
        pytest.param({'last_plan_s': 590.0}, id='plans end before the run'),
        pytest.param({'with_plans': False}, id='no plan'),
    ],
)
def test_missing_plans_do_not_establish_c1(run):
    assert judge(**run).verdict == Verdict.INVALID


def test_gap_just_below_t_gap_establishes_c1():
    # Removing the plans inside (300, 300.4) leaves 300.0 and 300.4 apart by 0.4 s.
    assert 0.4 < T_GAP_S
    assert judge(gap_at_s=300.0, gap_s=0.3).verdict == Verdict.PASS


def test_report_ends_with_the_outcome():
    report = str(C1Report(judge(with_plans=False)))
    assert report.splitlines()[-1] == 'V-08 C1: NOT established (invalid run)'
