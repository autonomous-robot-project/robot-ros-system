"""Unit tests of the V-07 criterion on synthetic State and Instrumentation streams."""

from dataclasses import dataclass, field

from robot_verification.check_v07 import (
    FAULT_TEST,
    N_EVENTS,
    T_SEQ_US,
    V07,
    V_R,
    ResponseRun,
    V07Report,
)
from robot_verification.recording import ControllerRecording, Sample, Verdict
from synthetic import command, instrumentation, periodic, state

INSTRUMENTATION_PERIOD_S = 0.1  # 10 Hz
STATE_PERIOD_S = 0.01  # 100 Hz
RESPONSE_US = 30_000  # a typical Δ


@dataclass
class SyntheticV07Run:
    """
    A V-07 recording from a controller and a sequencer that behave as configured.

    Event k (from 1) is activated at k * period_s and answered after its response; the State
    before each activation echoes TWIST(v_r, 0), and the one that ends it echoes TWIST(0, 0).
    """

    events: int = N_EVENTS
    period_s: float = 1.0
    responses_us: dict[int, int] = field(default_factory=dict)  # event -> Δ, else RESPONSE_US
    recording_starts_s: float = 0.0
    skipped_event: int | None = None  # its index never appears: the numbering jumps over it
    republished_stop_before: int | None = None  # TWIST(0, 0) in force before this event
    lost_activation_of: int | None = None  # every State with its code active is lost
    gap_at_s: float | None = None  # start of a gap in the Instrumentation
    gap_s: float = 0.0

    def recording(self) -> ControllerRecording:
        states = [s for s in self._states() if s.time_s >= self.recording_starts_s]
        end_s = (self.events + 1) * self.period_s
        samples = periodic(0.0, end_s, INSTRUMENTATION_PERIOD_S, self._instrumentation_at)
        kept = [s for s in samples if s.time_s >= self.recording_starts_s and not self._in_gap(s)]
        return ControllerRecording([], states, kept)

    def response_us(self, event: int) -> int:
        return self.responses_us.get(event, RESPONSE_US)

    def _states(self) -> list[Sample]:
        samples = []
        for event in range(1, self.events + 1):
            samples += self._states_of(event)
        return samples

    def _states_of(self, event: int) -> list[Sample]:
        activation_s = event * self.period_s
        answer_s = activation_s + self.response_us(event) * 1e-6
        cruise = command(0.0) if event == self.republished_stop_before else command(V_R)
        before = Sample(activation_s - STATE_PERIOD_S, state(cruise))
        active = Sample(activation_s, state(command(V_R), faults=FAULT_TEST))
        after = Sample(answer_s, state(command(0.0)))
        if event == self.lost_activation_of:
            return [before, after]
        return [before, active, after]

    def _instrumentation_at(self, t: float):
        answered = [e for e in range(1, self.events + 1) if self._answer_s(e) <= t]
        if not answered:
            return instrumentation()
        last = answered[-1]
        jumped = self.skipped_event is not None and last >= self.skipped_event
        index = last + 1 if jumped else last
        response_us = self.response_us(last)
        return instrumentation(test_event_index=index, test_event_response_us=response_us)

    def _answer_s(self, event: int) -> float:
        return event * self.period_s + self.response_us(event) * 1e-6

    def _in_gap(self, sample: Sample) -> bool:
        if self.gap_at_s is None:
            return False
        return self.gap_at_s <= sample.time_s < self.gap_at_s + self.gap_s


def judge(run: SyntheticV07Run) -> V07:
    return V07(ResponseRun(run.recording()))


def test_run_with_every_response_within_t_seq_passes():
    assert judge(SyntheticV07Run()).verdict is Verdict.PASS


def test_response_equal_to_t_seq_passes():
    run = SyntheticV07Run(responses_us={150: T_SEQ_US})
    assert judge(run).verdict is Verdict.PASS


def test_one_response_above_t_seq_fails():
    run = SyntheticV07Run(responses_us={150: T_SEQ_US + 1})
    assert judge(run).verdict is Verdict.FAIL


def test_fewer_than_n_events_is_invalid():
    run = SyntheticV07Run(events=N_EVENTS - 1)
    assert judge(run).verdict is Verdict.INVALID


def test_event_answered_before_the_recording_is_left_out():
    # The first event is seen in the bag, but its activation came before the recording.
    left_out = SyntheticV07Run(events=N_EVENTS, recording_starts_s=1.05)
    assert len(ResponseRun(left_out.recording()).get_recorded_events()) == N_EVENTS - 1
    assert judge(left_out).verdict is Verdict.INVALID
    enough = SyntheticV07Run(events=N_EVENTS + 1, recording_starts_s=1.05)
    assert judge(enough).verdict is Verdict.PASS


def test_slow_event_before_the_recording_does_not_fail():
    run = SyntheticV07Run(
        events=N_EVENTS + 1, responses_us={1: 2 * T_SEQ_US}, recording_starts_s=1.25
    )
    assert judge(run).verdict is Verdict.PASS


def test_skipped_index_is_invalid():
    run = SyntheticV07Run(events=N_EVENTS + 1, skipped_event=100)
    assert judge(run).verdict is Verdict.INVALID


def test_stop_in_force_before_an_event_is_invalid():
    # A TWIST(0, 0) already in force would end the event with a small Δ and no reaction.
    run = SyntheticV07Run(republished_stop_before=100)
    assert judge(run).verdict is Verdict.INVALID


def test_event_without_its_activation_in_the_state_is_invalid():
    run = SyntheticV07Run(lost_activation_of=100)
    assert judge(run).verdict is Verdict.INVALID


def test_gap_in_the_instrumentation_is_invalid():
    # Events every 3 s and a 1.5 s gap: no index is skipped, only the gap catches it.
    run = SyntheticV07Run(period_s=3.0, gap_at_s=301.5, gap_s=1.5)
    assert judge(run).verdict is Verdict.INVALID


def test_report_shows_the_statistics_of_the_response():
    run = SyntheticV07Run(responses_us={150: 80_000})
    report = str(V07Report(judge(run)))
    assert 'Δ max 80.0 ms' in report
    assert 'Δ median 30.0 ms' in report
    assert report.endswith('V-07: PASS')
