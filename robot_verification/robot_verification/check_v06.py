"""
V-06: command expiry (RNF-02), evaluated on the ros2 bag of the run.

Criticality: T1, evidence (robot-docs codigo.md).

Criterion: in each interruption, the expiry counter of the Instrumentation rises exactly once,
and the expiry delay measured in the controller (from the reception of the last command to the
cycle that expired it) is at most T_CMD + T_C (210 ms). TWIST(0, 0) stays in the echo until a new
command arrives.

The delay comes from the controller itself (ADR 028), measured from the reception of the command
as RNF-02 states; the bag times only delimit the interruptions.

Interruptions of the procedure last seconds, since step 4 waits for the stop in the echo. Shorter
gaps are pauses of the publisher (robot-docs#40); they are reported, not evaluated.

Validity of the run: a controller that ignores every command would echo TWIST(0, 0) all the time
and meet the criterion by accident. The run is valid only if the echo showed the published command
before each interruption, the procedure produced its thirty interruptions, and the Instrumentation
was recorded around each of them.

Usage: check_v06 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from dataclasses import dataclass
import math
import sys

from robot_verification.recording import ControllerRecording, Sample, Verdict

# Parameters of V-06.
T_CMD_S = 0.200
T_C_S = 0.010
MAX_EXPIRY_DELAY_S = T_CMD_S + T_C_S
EXPECTED_GAPS = 30  # procedure: steps 2 to 4, thirty times
PROCEDURE_GAP_MIN_S = 1.0  # shorter gaps are publisher pauses (~350 ms), not interruptions
INSTRUMENTATION_PERIOD_S = 0.100  # 10 Hz (ADR 025)
# The expiry reaches the bag at most one publication after it happens; a second period covers the
# link latency. An Instrumentation received this long after the last command must count it.
EXPIRY_REPORTED_WITHIN_S = MAX_EXPIRY_DELAY_S + 2 * INSTRUMENTATION_PERIOD_S

TWIST = 0  # Command.TWIST


@dataclass(frozen=True)
class CommandGap:
    """An interval longer than T_CMD without commands: from the last command to the next one."""

    last_command: Sample
    end_s: float  # the next command, or the end of the recording

    @property
    def start_s(self):
        return self.last_command.time_s


def is_stop(command):
    """Tell whether the command is TWIST(0, 0), the command in force after expiry."""
    parameters = (command.linear, command.angular, command.theta, command.x, command.y)
    return command.behavior == TWIST and all(value == 0.0 for value in parameters)


def is_same_command(echo, command):
    """Tell whether the echo carries the same behaviour and parameters as the command."""
    parameters = ('linear', 'angular', 'theta', 'x', 'y')
    return echo.behavior == command.behavior and all(
        math.isclose(getattr(echo, name), getattr(command, name)) for name in parameters
    )


def command_gaps(commands):
    """Every interval longer than T_CMD without commands, the one after the last included."""
    next_command_times_s = [c.time_s for c in commands[1:]] + [math.inf]
    return [
        CommandGap(command, next_s)
        for command, next_s in zip(commands, next_command_times_s, strict=True)
        if next_s - command.time_s > T_CMD_S
    ]


class V06:
    """The criterion of V-06 applied to the command gaps of a recording."""

    def __init__(self, recording):
        self.recording = recording
        all_gaps = command_gaps(recording.commands)
        self.gaps = [g for g in all_gaps if g.end_s - g.start_s > PROCEDURE_GAP_MIN_S]
        self.publisher_pauses = [g for g in all_gaps if g.end_s - g.start_s <= PROCEDURE_GAP_MIN_S]

    def expiries(self, gap):
        """Return how many times the expiry counter rose during the gap."""
        before, after = self._instrumentation_around(gap)
        return after.message.expiries - before.message.expiries

    def expiry_delay_s(self, gap):
        """Return the delay of the last expiry, as measured by the controller."""
        _, after = self._instrumentation_around(gap)
        return after.message.expiry_delay_us * 1e-6

    def stop_persisted(self, gap):
        """Check that TWIST(0, 0) stayed in the echo from the first stop to the end of the gap."""
        first_stop = self._first_stop(gap)
        if first_stop is None:
            return False
        held = [
            s
            for s in self.recording.states.between(gap.start_s, gap.end_s)
            if s.time_s >= first_stop.time_s
        ]
        return all(is_stop(s.message.echo) for s in held)

    def command_was_echoed(self, gap):
        """Check validity: the controller was applying the last command when commands stopped."""
        state = self.recording.states.at(gap.start_s)
        return state is not None and is_same_command(state.message.echo, gap.last_command.message)

    def instrumentation_covers(self, gap):
        """Check validity: the counter was recorded before the gap and after its expiry."""
        before = self.recording.instrumentation.at(gap.start_s)
        after = self.recording.instrumentation.at(gap.end_s)
        return (
            before is not None
            and after is not None
            and after.time_s >= gap.start_s + EXPIRY_REPORTED_WITHIN_S
        )

    def gap_passed(self, gap):
        return (
            self.expiries(gap) == 1
            and self.expiry_delay_s(gap) <= MAX_EXPIRY_DELAY_S
            and self.stop_persisted(gap)
        )

    @property
    def run_is_valid(self):
        return len(self.gaps) == EXPECTED_GAPS and all(
            self.command_was_echoed(gap) and self.instrumentation_covers(gap) for gap in self.gaps
        )

    @property
    def verdict(self):
        if not self.run_is_valid:
            return Verdict.INVALID
        return Verdict.PASS if all(self.gap_passed(gap) for gap in self.gaps) else Verdict.FAIL

    def report(self):
        lines = [
            f'V-06: {len(self.gaps)} command gaps ({EXPECTED_GAPS} expected), '
            f'{len(self.publisher_pauses)} publisher pauses not evaluated'
        ]
        for number, gap in enumerate(self.gaps, start=1):
            if not self.instrumentation_covers(gap):
                lines.append(f'  gap {number}: instrumentation not recorded around the gap')
                continue
            lines.append(
                f'  gap {number}: {self.expiries(gap)} expiries (1 expected), '
                f'expiry delay {self.expiry_delay_s(gap) * 1e3:.3f} ms '
                f'(max {MAX_EXPIRY_DELAY_S * 1e3:.0f} ms), '
                f'stop persisted: {self.stop_persisted(gap)}, '
                f'command echoed before: {self.command_was_echoed(gap)}'
            )
        lines.append(f'V-06: {self.verdict.name}')
        return '\n'.join(lines)

    def _instrumentation_around(self, gap):
        return (
            self.recording.instrumentation.at(gap.start_s),
            self.recording.instrumentation.at(gap.end_s),
        )

    def _first_stop(self, gap):
        states = self.recording.states.between(gap.start_s, gap.end_s)
        return next((s for s in states if is_stop(s.message.echo)), None)


def main():
    if len(sys.argv) != 2:
        print('usage: check_v06 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    v06 = V06(ControllerRecording.from_bag(sys.argv[1]))
    print(v06.report())
    sys.exit(v06.verdict.value)


if __name__ == '__main__':
    main()
