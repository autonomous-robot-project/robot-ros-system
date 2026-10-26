"""
V-02: cognizant failure (RF-02), evaluated on the ros2 bag recorded by run_v02.

Criterion: the failure code is active during each stall and inactive outside them, including the
start, with no clear command. After the infinite-loop injection, the State brings the code
"restart by watchdog" (ADR 029).

The stalls are the windows announced by run_v02 on the marker topic ("stall", "release"), not the
measured speed, which is what the controller itself uses to detect the failure. Between an
announcement and the hand on (or off) the wheel there is the reaction of the operator, so each
window is judged with that tolerance:
- active: by the stall + reaction + detection latency (t* + T_f, V-02 step 8), and from then on
  until the release;
- inactive: by the release + reaction + T_f, and from then on until the next stall;
- inactive from the first state until the first stall (the start).

Validity of the run: the procedure produced its three stalls and the injection; the echo showed
TWIST(v_r, 0) when each stall was announced, so the controller was applying the command; the
measured speed of the locked wheel reached zero in each window, so the operator did lock it; and
the State was recorded with no hole until the injection.

Usage: check_v02 <bag directory>. Exit code: 0 pass, 1 fail, 2 invalid run.
"""

from dataclasses import dataclass
import math
import sys

from robot_verification.recording import Sample, Stream, Verdict, read_bag

# Parameters of V-02.
V_R = 0.2  # m/s, commanded in the trial
E_F = 0.5
T_F_S = 1.0  # 2 x t_s of design
T_C_S = 0.010
WHEEL_TRAVEL_PER_EDGE = math.pi * 0.065 / 20  # s (m)
T_STAR_S = WHEEL_TRAVEL_PER_EDGE / ((1 - E_F) * V_R)  # stalled wheel reads below (1 - e_f) v_r
DETECTION_LATENCY_S = T_STAR_S + T_F_S + T_C_S  # V-02 step 8, plus one cycle

# Conditions of the run.
EXPECTED_STALLS = 3
OPERATOR_REACTION_S = 0.5  # from an announcement to the hand on (or off) the wheel
LINK_DELAY_S = 0.05  # from the controller to the bag
MAX_STATE_GAP_S = 0.1  # longer is a hole in the recording, or the watchdog restart
ACTIVE_BY_S = OPERATOR_REACTION_S + DETECTION_LATENCY_S + LINK_DELAY_S
INACTIVE_BY_S = OPERATOR_REACTION_S + T_F_S + LINK_DELAY_S

# Codes of State.msg, checked against the generated message when a bag is read.
FAULT_COGNIZANT = 1
FAULT_WATCHDOG_RESTART = 4
TWIST = 0  # Command.TWIST

STATE_TOPIC = '/controller/state'
INSTRUMENTATION_TOPIC = '/controller/instrumentation'
MARKER_TOPIC = '/verification/v02_marker'
STALL, RELEASE, INJECT = 'stall', 'release', 'inject'


@dataclass(frozen=True)
class StallWindow:
    """A stall as announced by run_v02: from "stall" to "release"."""

    stall_s: float
    release_s: float


def cognizant_failure(state):
    return bool(state.message.faults & FAULT_COGNIZANT)


def watchdog_restart(state):
    return bool(state.message.faults & FAULT_WATCHDOG_RESTART)


def echoes_trial_command(state):
    echo = state.message.echo
    return echo.behavior == TWIST and math.isclose(echo.linear, V_R) and echo.angular == 0.0


class Recording:
    """The states, instrumentation and markers of a V-02 run, in reception order."""

    def __init__(self, states, instrumentation, markers):
        self.states = Stream(states)
        self.instrumentation = Stream(instrumentation)
        self.markers = Stream(markers)  # the message is the name of the marker

    @classmethod
    def from_bag(cls, path):
        # ROS imports here, so the criterion can be tested without ROS.
        from robot_interfaces.msg import Instrumentation, State
        from std_msgs.msg import String

        assert State.FAULT_COGNIZANT == FAULT_COGNIZANT
        assert State.FAULT_WATCHDOG_RESTART == FAULT_WATCHDOG_RESTART
        streams = read_bag(
            path,
            {STATE_TOPIC: State, INSTRUMENTATION_TOPIC: Instrumentation, MARKER_TOPIC: String},
        )
        markers = [Sample(m.time_s, m.message.data) for m in streams[MARKER_TOPIC]]
        return cls(streams[STATE_TOPIC], streams[INSTRUMENTATION_TOPIC], markers)

    def marker_times(self, name):
        return [marker.time_s for marker in self.markers if marker.message == name]

    def stall_windows(self):
        stalls, releases = self.marker_times(STALL), self.marker_times(RELEASE)
        return [StallWindow(s, r) for s, r in zip(stalls, releases, strict=False)]

    def injection_s(self):
        injections = self.marker_times(INJECT)
        return injections[0] if injections else None


class V02:
    """The criterion of V-02 applied to the announced stalls of a recording."""

    def __init__(self, recording):
        self.recording = recording
        self.windows = recording.stall_windows()
        self.injection_s = recording.injection_s()

    # --- Criterion ---

    def quiet_at_start(self):
        """Check that the start, until the first stall, shows no failure."""
        start = self.recording.states.between(-math.inf, self.windows[0].stall_s)
        return bool(start) and not any(cognizant_failure(s) for s in start)

    def active_during(self, window):
        """Check that the failure is active by its deadline and until the release."""
        held = self.recording.states.between(window.stall_s + ACTIVE_BY_S, window.release_s)
        return bool(held) and all(cognizant_failure(s) for s in held)

    def inactive_after(self, window, next_stall_s):
        """Check that the failure is inactive by its deadline and until the next stall."""
        after = self.recording.states.between(window.release_s + INACTIVE_BY_S, next_stall_s)
        return bool(after) and not any(cognizant_failure(s) for s in after)

    def stall_passed(self, index):
        window = self.windows[index]
        is_last = index + 1 == len(self.windows)
        next_stall_s = self.injection_s if is_last else self.windows[index + 1].stall_s
        return self.active_during(window) and self.inactive_after(window, next_stall_s)

    def watchdog_restart_reported(self):
        """Check that the first state after the restart brings the code of the watchdog."""
        first_after_restart = self._first_state_after_restart()
        return first_after_restart is not None and watchdog_restart(first_after_restart)

    # --- Validity of the run ---

    def command_echoed(self, window):
        state = self.recording.states.at(window.stall_s)
        return state is not None and echoes_trial_command(state)

    def wheel_locked(self, window):
        """Check that the measured speed of the left wheel reached zero in the window."""
        samples = self.recording.instrumentation.between(window.stall_s, window.release_s)
        return any(s.message.left_wheel_speed == 0.0 for s in samples)

    def recorded_without_holes(self):
        states = self.recording.states.between(-math.inf, self.injection_s)
        return not _holes(states)

    @property
    def run_is_valid(self):
        return (
            len(self.windows) == EXPECTED_STALLS
            and self.injection_s is not None
            and all(self.command_echoed(w) and self.wheel_locked(w) for w in self.windows)
            and self.recorded_without_holes()
        )

    # --- Verdict ---

    @property
    def verdict(self):
        if not self.run_is_valid:
            return Verdict.INVALID
        passed = (
            self.quiet_at_start()
            and all(self.stall_passed(index) for index in range(len(self.windows)))
            and self.watchdog_restart_reported()
        )
        return Verdict.PASS if passed else Verdict.FAIL

    def report(self):
        lines = [
            f'V-02: {len(self.windows)} stalls ({EXPECTED_STALLS} expected), '
            f'injection recorded: {self.injection_s is not None}',
            f'  tolerances: active within {ACTIVE_BY_S:.2f} s of each stall, '
            f'inactive within {INACTIVE_BY_S:.2f} s of each release',
        ]
        if self.injection_s is not None:
            lines.append(f'  state recorded without holes: {self.recorded_without_holes()}')
        if self.windows:
            lines.append(f'  start without failure: {self.quiet_at_start()}')
        for index, window in enumerate(self.windows):
            lines.append(
                f'  stall {index + 1}: command echoed: {self.command_echoed(window)}, '
                f'wheel locked: {self.wheel_locked(window)}, passed: {self.stall_passed(index)}'
            )
        lines.append(f'  watchdog restart reported: {self.watchdog_restart_reported()}')
        lines.append(f'V-02: {self.verdict.name}')
        return '\n'.join(lines)

    def _first_state_after_restart(self):
        """Return the first state after the injection that follows a hole: the restart."""
        if self.injection_s is None:
            return None
        restarts = [
            after for _, after in _holes(self.recording.states) if after.time_s > self.injection_s
        ]
        return restarts[0] if restarts else None


def _holes(states):
    """Return each pair of consecutive states farther apart than MAX_STATE_GAP_S."""
    states = list(states)
    return [
        (before, after)
        for before, after in zip(states, states[1:], strict=False)
        if after.time_s - before.time_s > MAX_STATE_GAP_S
    ]


def main():
    if len(sys.argv) != 2:
        print('usage: check_v02 <bag directory>')
        sys.exit(Verdict.INVALID.value)
    v02 = V02(Recording.from_bag(sys.argv[1]))
    print(v02.report())
    sys.exit(v02.verdict.value)


if __name__ == '__main__':
    main()
