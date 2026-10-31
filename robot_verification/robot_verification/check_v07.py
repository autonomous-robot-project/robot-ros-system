"""
V-07: response of the sequencer (RNF-03), judged on the ros2 bag recorded by run_v07.

Criticality: T1, evidence (robot-docs codigo.md).

What: no test event has a response Δ above T_seq; reports the maximum, the median and the 99th
percentile of Δ.
How: in test mode the controller activates FAULT_TEST every T_ev, ends it at the arrival of a
TWIST(0, 0) and publishes, in the Instrumentation, the index of each answered event and its Δ:
the arrival of the TWIST(0, 0) minus the t_k of the cycle that activated the code, both in the
clock of the ESP32 (ADR 049). Each index is one event; the State shows its activation. The script
judges only the Δ that the controller measured; it never times the sequencer itself.
Why: V-07 (robot-docs verifications/V-07). The limit holds P(Δ > T_seq) below 1 % with 95 %
confidence when none of n = 300 events exceeds it (Wilks, 1941).

Criterion (V-07, RNF-03): none of the n events has Δ > T_seq, which bounds P(Δ > T_seq) to 1 %
with 95 % confidence. Report the maximum, the median and the 99th percentile of Δ.

Validity of the run: at least n events whose activation is in the recording (an event answered
before the recording started is left out); the indices are contiguous, so no event and no restart
of the ESP32 went unseen; the last State before each activation echoes TWIST(v_r, 0) with the code
off, so no Δ comes from a TWIST(0, 0) published before the event; and the Instrumentation has no
gap that a restart would leave.

Usage: check_v07 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import statistics
import sys
from typing import TYPE_CHECKING

from robot_verification.check_v05 import MAX_INSTRUMENTATION_GAP_S
from robot_verification.recording import (
    ControllerRecording,
    Sample,
    Verdict,
    is_close_float32,
)

if TYPE_CHECKING:
    from robot_interfaces.msg import Command, Instrumentation, State

# Parameters of V-07.
T_SEQ_US: int = 100_000  # maximum response of the sequencer: one scan period of the LiDAR
N_EVENTS: int = 300  # n >= ln(1 - 0.95) / ln(0.99) = 298.1 (Wilks, 1941)
V_R: float = 0.2  # m/s, commanded before each event

# Values of the messages, checked against them when a bag is read.
FAULT_TEST: int = 16  # State.FAULT_TEST
TWIST: int = 0  # Command.TWIST

PERCENTILES: int = 100  # statistics.quantiles in hundredths
P99_INDEX: int = 98  # the 99th of the 99 cut points
US_PER_MS: int = 1000


@dataclass(frozen=True)
class ResponseEvent:
    """One answered test event, as the Instrumentation reports it."""

    index: int
    response_us: int
    seen_s: float  # reception of the first Instrumentation that carries it


@dataclass(frozen=True)
class Activation:
    """An activation of FAULT_TEST in the State, with the State received just before it."""

    time_s: float
    previous: State


def is_test_code_active(state: State) -> bool:
    return bool(state.faults & FAULT_TEST)


def is_cruise_echo(echo: Command) -> bool:
    """Tell whether the echo is TWIST(v_r, 0), the command in force before each event."""
    return echo.behavior == TWIST and is_close_float32(echo.linear, V_R) and echo.angular == 0.0


def is_cruise_activation(activation: Activation | None) -> bool:
    """Tell whether an activation was recorded, with TWIST(v_r, 0) in the State before it."""
    return activation is not None and is_cruise_echo(activation.previous.echo)


def find_events(samples: list[Sample[Instrumentation]]) -> list[ResponseEvent]:
    """Return each answered event once, at the first sample that carries its index."""
    events: list[ResponseEvent] = []
    for sample in samples:
        index = sample.message.test_event_index
        is_new = index > 0 and (not events or index != events[-1].index)
        if is_new:
            events.append(
                ResponseEvent(index, sample.message.test_event_response_us, sample.time_s)
            )
    return events


def find_activations(samples: list[Sample[State]]) -> list[Activation]:
    """Return each State in which FAULT_TEST turns on; the first State has no known previous."""
    return [
        Activation(current.time_s, previous.message)
        for previous, current in itertools.pairwise(samples)
        if not is_test_code_active(previous.message) and is_test_code_active(current.message)
    ]


def find_last_activation_between(
    activations: list[Activation], start_s: float, end_s: float
) -> Activation | None:
    inside = [a for a in activations if start_s < a.time_s < end_s]
    return inside[-1] if inside else None


def pair_activations(
    events: list[ResponseEvent], activations: list[Activation]
) -> list[Activation | None]:
    """
    Return the activation of each event, or None where it is not in the recording.

    It is the last activation after the previous event was seen and before this one is.
    """
    window_starts_s = [float('-inf')] + [event.seen_s for event in events[:-1]]
    return [
        find_last_activation_between(activations, start_s, event.seen_s)
        for start_s, event in zip(window_starts_s, events, strict=True)
    ]


class ResponseRun:
    """The test events of a recording and the activations of FAULT_TEST that started them."""

    def __init__(self, recording: ControllerRecording):
        self.instrumentation: list[Sample[Instrumentation]] = list(recording.instrumentation)
        self.events: list[ResponseEvent] = find_events(self.instrumentation)
        self.activations: list[Activation] = find_activations(list(recording.states))
        # Aligned with self.events.
        self.activations_of_events: list[Activation | None] = pair_activations(
            self.events, self.activations
        )

    @classmethod
    def from_bag(cls, path: str) -> ResponseRun:
        # ROS imports here, so the criterion can be tested without ROS.
        from robot_interfaces.msg import Command, State

        assert State.FAULT_TEST == FAULT_TEST
        assert Command.TWIST == TWIST
        return cls(ControllerRecording.from_bag(path))

    def measure_max_response_us(self) -> int:
        return max(self.get_responses_us())

    def measure_median_response_us(self) -> float:
        return statistics.median(self.get_responses_us())

    def measure_p99_response_us(self) -> float:
        responses_us = self.get_responses_us()
        quantiles = statistics.quantiles(responses_us, n=PERCENTILES, method='inclusive')
        return quantiles[P99_INDEX]

    def get_responses_us(self) -> list[int]:
        return [event.response_us for event in self.get_recorded_events()]

    def get_recorded_events(self) -> list[ResponseEvent]:
        """Return the events whose activation is in the recording."""
        pairs = zip(self.events, self.activations_of_events, strict=True)
        return [event for event, activation in pairs if activation is not None]

    def measure_longest_instrumentation_gap_s(self) -> float:
        pairs = itertools.pairwise(self.instrumentation)
        return max((after.time_s - before.time_s for before, after in pairs), default=0.0)


class V07:
    """
    The criterion and the validity conditions of V-07, and its verdict.

    Valid run: n recorded events, contiguous indices, the cruise echo before every activation, and
    no gap in the Instrumentation that a restart of the ESP32 would leave.
    """

    def __init__(self, run: ResponseRun):
        self.run: ResponseRun = run

    # --- Verdict ---

    @property
    def verdict(self) -> Verdict:
        if not self.run_is_valid:
            return Verdict.INVALID
        return Verdict.PASS if self.did_every_response_meet_t_seq() else Verdict.FAIL

    @property
    def run_is_valid(self) -> bool:
        return (
            self.has_n_events()
            and self.are_indices_contiguous()
            and self.was_cruising_before_every_event()
            and self.was_recorded_without_restart()
        )

    # --- Criterion ---

    def did_every_response_meet_t_seq(self) -> bool:
        return self.run.measure_max_response_us() <= T_SEQ_US

    # --- Validity of the run ---

    def has_n_events(self) -> bool:
        return len(self.run.get_recorded_events()) >= N_EVENTS

    def are_indices_contiguous(self) -> bool:
        indices = [event.index for event in self.run.events]
        return all(after == before + 1 for before, after in itertools.pairwise(indices))

    def was_cruising_before_every_event(self) -> bool:
        """
        Tell whether every event has its activation, preceded by TWIST(v_r, 0).

        Only the first may lack it, because it may have started before the recording.
        """
        activations = list(self.run.activations_of_events)
        if activations and activations[0] is None:
            activations.pop(0)
        return all(is_cruise_activation(activation) for activation in activations)

    def was_recorded_without_restart(self) -> bool:
        return self.run.measure_longest_instrumentation_gap_s() <= MAX_INSTRUMENTATION_GAP_S


class V07Report:
    """The report of V-07: the statistics of Δ and the verdict of each check."""

    def __init__(self, verification: V07):
        self.verification: V07 = verification
        self.run: ResponseRun = verification.run

    def __str__(self) -> str:
        recorded = len(self.run.get_recorded_events())
        lines = [f'V-07: {len(self.run.events)} test events, {recorded} with their activation']
        lines += self._format_validity_lines()
        if recorded >= 2:
            lines += self._format_response_lines()
        lines.append(f'V-07: {self.verification.verdict.name}')
        return '\n'.join(lines)

    def _format_validity_lines(self) -> list[str]:
        v = self.verification
        gap_s = self.run.measure_longest_instrumentation_gap_s()
        return [
            f'  events with activation >= {N_EVENTS}, passed: {v.has_n_events()}',
            f'  contiguous indices, passed: {v.are_indices_contiguous()}',
            f'  TWIST({V_R}, 0) before each activation, passed: '
            f'{v.was_cruising_before_every_event()}',
            f'  longest Instrumentation gap {gap_s:.3f} s '
            f'(max {MAX_INSTRUMENTATION_GAP_S:.1f} s), passed: {v.was_recorded_without_restart()}',
        ]

    def _format_response_lines(self) -> list[str]:
        max_ms = self.run.measure_max_response_us() / US_PER_MS
        median_ms = self.run.measure_median_response_us() / US_PER_MS
        p99_ms = self.run.measure_p99_response_us() / US_PER_MS
        return [
            f'  Δ max {max_ms:.1f} ms (T_seq {T_SEQ_US / US_PER_MS:.0f} ms), passed: '
            f'{self.verification.did_every_response_meet_t_seq()}',
            f'  Δ median {median_ms:.1f} ms, p99 {p99_ms:.1f} ms',
        ]


def main() -> None:
    if len(sys.argv) != 2:
        print('usage: check_v07 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    verification = V07(ResponseRun.from_bag(sys.argv[1]))
    print(V07Report(verification))
    sys.exit(verification.verdict.value)


if __name__ == '__main__':
    main()
