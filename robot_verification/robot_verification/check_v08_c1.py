"""
V-08, condition C1: plans delivered during the whole run, judged on the ros2 bag of a C1 run.

Criticality: T1, evidence (robot-docs codigo.md).

What: tells whether C1 was established; it does not judge V-08 itself. The criteria of a C1 run
are those of V-05 and V-07, read by check_v05 and check_v07 on the same bag.
How: the span of the run is that of the Instrumentation, which the controller publishes every
period while the bag records. The plans are the Path on /plan, which the planner publishes at
each response to the sequencer; the sequencer asks again at each response. C1 holds when the first
plan arrives at most T_gap after the first Instrumentation, consecutive plans are at most T_gap
apart, and the last Instrumentation arrives at most T_gap after the last plan. The condition is
established before the recording starts (step 1 of V-08), so the first plan needs no extra margin.
Why: V-08 (robot-docs verifications/V-08, nota 0019).

Validity of the run (V-08): "Em C1, o plano é entregue ao longo de toda a execução; sem isso, a
condição não foi estabelecida."

Usage: check_v08_c1 <bag directory>. Exit code: 0 C1 established, 2 not established (invalid run).
"""

from __future__ import annotations

from collections.abc import Iterable
import itertools
import math
import sys
from typing import TYPE_CHECKING

from robot_verification.check_v04 import PLAN_TOPIC
from robot_verification.recording import (
    INSTRUMENTATION_TOPIC,
    Sample,
    Stream,
    Verdict,
    read_bag,
)

if TYPE_CHECKING:
    from nav_msgs.msg import Path
    from robot_interfaces.msg import Instrumentation

# Longest time without a new plan: T_plan + 0.5 s. In the C1 run with T_plan = 1 s of robot-docs
# nota 0019 the plans came at most 1.006 s apart, and the first 1.151 s after the first
# Instrumentation.
T_GAP_S: float = 1.5


class PlanDeliveryRun:
    """The plans of a C1 run and the span of the run, given by the Instrumentation."""

    def __init__(
        self,
        plans: Iterable[Sample[Path]],
        instrumentation: Iterable[Sample[Instrumentation]],
    ):
        self.plans: Stream[Path] = Stream(plans)
        self.instrumentation: Stream[Instrumentation] = Stream(instrumentation)

    @classmethod
    def from_bag(cls, path: str) -> PlanDeliveryRun:
        # ROS imports here, so the criteria can be tested without ROS.
        from nav_msgs.msg import Path
        from robot_interfaces.msg import Instrumentation

        streams = read_bag(path, {PLAN_TOPIC: Path, INSTRUMENTATION_TOPIC: Instrumentation})
        return cls(streams[PLAN_TOPIC], streams[INSTRUMENTATION_TOPIC])

    def has_plans(self) -> bool:
        return len(self.plans) > 0 and len(self.instrumentation) > 0

    def measure_wait_for_first_plan_s(self) -> float:
        """Return the time from the start of the run to the first plan; infinite without plans."""
        if not self.has_plans():
            return math.inf
        return self.plans[0].time_s - self.instrumentation[0].time_s

    def measure_longest_gap_between_plans_s(self) -> float:
        """Return the longest time between consecutive plans; zero with fewer than two."""
        return max(
            (after.time_s - before.time_s for before, after in itertools.pairwise(self.plans)),
            default=0.0,
        )

    def measure_time_after_last_plan_s(self) -> float:
        """Return the time from the last plan to the end of the run; infinite without plans."""
        if not self.has_plans():
            return math.inf
        return self.instrumentation[-1].time_s - self.plans[-1].time_s


class C1Establishment:
    """Whether condition C1 of V-08 held: a plan at most every T_gap from start to end."""

    def __init__(self, run: PlanDeliveryRun):
        self.run: PlanDeliveryRun = run

    @property
    def verdict(self) -> Verdict:
        return Verdict.PASS if self.was_established else Verdict.INVALID

    @property
    def was_established(self) -> bool:
        return (
            self.did_first_plan_arrive_in_t_gap()
            and self.were_plans_at_most_t_gap_apart()
            and self.did_plans_last_until_the_end()
        )

    def did_first_plan_arrive_in_t_gap(self) -> bool:
        return self.run.measure_wait_for_first_plan_s() <= T_GAP_S

    def were_plans_at_most_t_gap_apart(self) -> bool:
        return self.run.measure_longest_gap_between_plans_s() <= T_GAP_S

    def did_plans_last_until_the_end(self) -> bool:
        return self.run.measure_time_after_last_plan_s() <= T_GAP_S


class C1Report:
    """The report of C1: the plans, the three times against T_gap and the outcome."""

    def __init__(self, establishment: C1Establishment):
        self.establishment: C1Establishment = establishment
        self.run: PlanDeliveryRun = establishment.run

    def __str__(self) -> str:
        e = self.establishment
        longest_gap_s = self.run.measure_longest_gap_between_plans_s()
        outcome = 'established' if e.was_established else 'NOT established (invalid run)'
        return '\n'.join(
            [
                f'V-08 C1: {len(self.run.plans)} plans (T_gap {T_GAP_S:.3f} s)',
                f'  first plan after {self.run.measure_wait_for_first_plan_s():.3f} s, '
                f'passed: {e.did_first_plan_arrive_in_t_gap()}',
                f'  longest gap between plans {longest_gap_s:.3f} s, '
                f'passed: {e.were_plans_at_most_t_gap_apart()}',
                f'  end of the run {self.run.measure_time_after_last_plan_s():.3f} s after the '
                f'last plan, passed: {e.did_plans_last_until_the_end()}',
                f'V-08 C1: {outcome}',
            ]
        )


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v08_c1 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    establishment = C1Establishment(PlanDeliveryRun.from_bag(sys.argv[1]))
    print(C1Report(establishment))
    sys.exit(establishment.verdict.value)


if __name__ == '__main__':
    main()
