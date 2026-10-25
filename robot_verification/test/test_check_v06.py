"""Unit tests of the V-06 criterion on synthetic command and state streams."""

from types import SimpleNamespace

from robot_verification.check_v06 import EXPECTED_GAPS, V06, Recording, Sample, Verdict

COMMAND_PERIOD_S = 0.05  # 20 Hz
STATE_PERIOD_S = 0.01  # 100 Hz
RUN_S = 1.0  # commands published before each interruption
PAUSE_S = 2.0  # no command after each interruption
INSTRUMENTATION_PERIOD_S = 0.1  # 10 Hz


def command(linear=0.2):
    return SimpleNamespace(behavior=0, linear=linear, angular=0.0, theta=0.0, x=0.0, y=0.0)


def state(echo):
    return SimpleNamespace(faults=0, reflexes=0, echo=echo)


def instrumentation(expiries, expiry_delay_s):
    return SimpleNamespace(expiries=expiries, expiry_delay_us=round(expiry_delay_s * 1e6))


def run(
    expiry_delay_s=0.205,
    expiries_per_gap=1,
    gaps=EXPECTED_GAPS,
    ignore_commands=False,
    relapse=False,
    record_instrumentation=True,
):
    """
    Streams of a V-06 run.

    expiry_delay_s: delay from the last command to its expiry, as the controller measures it; the
    echo shows the stop from then on.
    expiries_per_gap: how many times the expiry counter rises in each gap.
    ignore_commands: the echo is always TWIST(0, 0), as in a controller ignoring the commands.
    relapse: the echo goes back to the command 0.5 s after the stop, with no new command.
    record_instrumentation: whether the bag has the Instrumentation topic.
    """
    commands, states, expiry_times = [], [], []
    start = 0.0
    for _ in range(gaps):
        times = [start + k * COMMAND_PERIOD_S for k in range(int(RUN_S / COMMAND_PERIOD_S))]
        commands += [Sample(t, command()) for t in times]
        last = times[-1]
        end = last + PAUSE_S
        expiry_times += [last + expiry_delay_s] * expiries_per_gap
        t = start
        while t < end:
            stopped = ignore_commands or (last + expiry_delay_s <= t)
            if relapse and t >= last + expiry_delay_s + 0.5:
                stopped = False
            states.append(Sample(t, state(command(0.0) if stopped else command())))
            t += STATE_PERIOD_S
        start = end
    return (
        commands,
        states,
        instrumentation_stream(start, expiry_times, expiry_delay_s, record_instrumentation),
    )


def instrumentation_stream(end_s, expiry_times, expiry_delay_s, recorded):
    """Return the Instrumentation at 10 Hz, whose counter accumulates the expiries so far."""
    if not recorded:
        return []
    samples = []
    t = 0.0
    while t < end_s:
        expired = sum(1 for e in expiry_times if e <= t)
        delay_s = expiry_delay_s if expired else 0.0
        samples.append(Sample(t, instrumentation(expired, delay_s)))
        t += INSTRUMENTATION_PERIOD_S
    return samples


def test_pass_when_expiry_within_limit():
    v06 = V06(Recording(*run(expiry_delay_s=0.205)))
    assert v06.verdict == Verdict.PASS
    assert len(v06.gaps) == EXPECTED_GAPS


def test_fail_when_expiry_too_late():
    assert V06(Recording(*run(expiry_delay_s=0.215))).verdict == Verdict.FAIL


def test_fail_when_counter_does_not_rise():
    assert V06(Recording(*run(expiries_per_gap=0))).verdict == Verdict.FAIL


def test_fail_when_stop_does_not_persist():
    assert V06(Recording(*run(relapse=True))).verdict == Verdict.FAIL


def test_invalid_when_controller_ignores_commands():
    assert V06(Recording(*run(ignore_commands=True))).verdict == Verdict.INVALID


def test_invalid_with_wrong_number_of_gaps():
    assert V06(Recording(*run(gaps=2))).verdict == Verdict.INVALID


def test_invalid_without_instrumentation():
    assert V06(Recording(*run(record_instrumentation=False))).verdict == Verdict.INVALID


def test_publisher_pause_is_not_an_interruption():
    commands, states, instrumentation_samples = run()
    paused = [c for c in commands if not 0.3 < c.time_s < 0.65]  # a 0.45 s gap in the first run
    v06 = V06(Recording(paused, states, instrumentation_samples))
    assert v06.verdict == Verdict.PASS
    assert len(v06.publisher_pauses) == 1
