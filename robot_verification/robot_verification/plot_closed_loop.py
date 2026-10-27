"""
Charts of the closed-loop run (robot-docs nota 0011) from its ros2 bag.

Criticality: T2, product (robot-docs codigo.md).

The whole schedule and a zoom on the stop, with the expected and limit times of
measure_closed_loop.

Usage: plot_closed_loop <bag directory> <output prefix>. Writes <prefix>_completo.png and
<prefix>_parada.png.
"""

from __future__ import annotations

import sys

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from robot_verification.measure_closed_loop import (
    STOP_EXPECTED_S,
    STOP_FROM,
    STOP_LIMIT_S,
    ClosedLoopRun,
    Wheel,
    get_wheel_speed,
    has_cognizant_failure,
)
from robot_verification.recording import ControllerRecording

LABELS: dict[Wheel, str] = {Wheel.LEFT: 'roda esquerda', Wheel.RIGHT: 'roda direita'}
COLORS: dict[Wheel, str] = {Wheel.LEFT: '#2a78d6', Wheel.RIGHT: '#e07b39'}
COMMAND, FAULT = '#52514e', '#c8423b'
MUTED, GRID, SURFACE = '#52514e', '#e4e3df', '#fcfcfb'

plt.rcParams.update(
    {
        'figure.facecolor': SURFACE,
        'axes.facecolor': SURFACE,
        'axes.edgecolor': GRID,
        'axes.labelcolor': MUTED,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'axes.grid': True,
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'font.size': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
    }
)


def plot_command(
    axis: Axes, recording: ControllerRecording, origin_s: float, start_s: float, end_s: float
) -> None:
    commands = recording.commands.between(start_s, end_s)
    axis.step(
        [c.time_s - origin_s for c in commands],
        [c.message.linear for c in commands],
        where='post',
        color=COMMAND,
        linewidth=1.0,
        linestyle='--',
        label='comando',
    )


def plot_wheel_speed(
    axis: Axes,
    recording: ControllerRecording,
    wheel: Wheel,
    origin_s: float,
    start_s: float,
    end_s: float,
) -> None:
    samples = recording.instrumentation.between(start_s, end_s)
    axis.plot(
        [s.time_s - origin_s for s in samples],
        [get_wheel_speed(s, wheel) for s in samples],
        color=COLORS[wheel],
        linewidth=1.2,
        marker='.',
        markersize=3,
        label=LABELS[wheel],
    )


def plot_false_alarms(
    axis: Axes, recording: ControllerRecording, origin_s: float, start_s: float, end_s: float
) -> None:
    states = recording.states.between(start_s, end_s)
    faults = [s for s in states if has_cognizant_failure(s)]
    if faults:
        axis.plot(
            [s.time_s - origin_s for s in faults],
            [0.0] * len(faults),
            '|',
            color=FAULT,
            markersize=10,
            label='FAULT_COGNIZANT',
        )


def plot_stop_marks(axis: Axes) -> None:
    """Draw the expected and limit times of the stop, from the stop command."""
    for at_s, label in ((STOP_EXPECTED_S, 'esperado'), (STOP_LIMIT_S, 'limite')):
        axis.axvline(at_s, color=MUTED, linewidth=0.8, linestyle=':')
        axis.text(at_s, STOP_FROM / 2, label, color=MUTED, fontsize=7, rotation=90, ha='right')


def plot_speeds(
    axis: Axes, recording: ControllerRecording, origin_s: float, start_s: float, end_s: float
) -> None:
    """Draw the commanded and measured speeds between start_s and end_s, times from origin_s."""
    plot_command(axis, recording, origin_s, start_s, end_s)
    for wheel in Wheel:
        plot_wheel_speed(axis, recording, wheel, origin_s, start_s, end_s)
    plot_false_alarms(axis, recording, origin_s, start_s, end_s)
    axis.set_xlabel('tempo (s)')
    axis.set_ylabel('velocidade (m/s)')
    axis.legend(loc='best')


def plot_whole(run: ClosedLoopRun, prefix: str) -> None:
    start_s, end_s = run.steps[0].start_s, run.steps[-1].end_s
    figure, axis = plt.subplots(figsize=(9, 3.6))
    plot_speeds(axis, run.recording, start_s, start_s, end_s)
    axis.set_title('Malha fechada, rodas suspensas', loc='left')
    figure.tight_layout()
    figure.savefig(f'{prefix}_completo.png', dpi=150)


def plot_stop(run: ClosedLoopRun, prefix: str) -> None:
    stop_s = run.find_stop_step().start_s
    figure, axis = plt.subplots(figsize=(6, 3.2))
    plot_speeds(axis, run.recording, stop_s, stop_s - 1.0, stop_s + 2.0)
    plot_stop_marks(axis)
    axis.set_title('Parada: velocidade zero após T_vel', loc='left')
    figure.tight_layout()
    figure.savefig(f'{prefix}_parada.png', dpi=150)


def main() -> None:
    if len(sys.argv) != 3:
        print('usage: plot_closed_loop <bag directory> <output prefix>')
        sys.exit(2)
    run = ClosedLoopRun(ControllerRecording.from_bag(sys.argv[1]))
    plot_whole(run, sys.argv[2])
    plot_stop(run, sys.argv[2])


if __name__ == '__main__':
    main()
