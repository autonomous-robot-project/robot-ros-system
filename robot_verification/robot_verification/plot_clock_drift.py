"""
Charts of the clock drift between the ESP32 and the RPi (ADR 027) from the bag of V-07.

Criticality: T2, product (robot-docs codigo.md).

The phase of each stamp over time, with the lowest per window, a staircase whose slope is the
drift; the histogram of the change of the low between consecutive windows, where the flatness of
the plateaus and the steps separate and JUMP_THRESHOLD_NS is chosen; the steps over time, whose
spread is the error of the synchronization. The numbers come from measure_clock_drift.

Usage: plot_clock_drift <bag directory> <output prefix>. Writes <prefix>_offset.png,
<prefix>_janelas.png and <prefix>_saltos.png.
"""

from __future__ import annotations

import itertools
import statistics
import sys

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from robot_verification.measure_clock_drift import (
    JUMP_THRESHOLD_NS,
    NS_PER_S,
    NS_PER_US,
    ClockDriftRun,
)
from robot_verification.recording import ControllerRecording

SERIES, MUTED, GRID, SURFACE = '#2a78d6', '#52514e', '#e4e3df', '#fcfcfb'
FIGURE_SIZE = (6.0, 3.0)  # in, the width of a column of the monograph
DPI = 150
HISTOGRAM_BINS = 60
SMALLEST_CHANGE_US = 0.1  # left end of the logarithmic axis; a zero change is put here

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


def get_times_s(run: ClockDriftRun) -> list[float]:
    """Return the time of each stamp from the first, in s."""
    return [(stamp - run.stamps_ns[0]) / NS_PER_S for stamp in run.stamps_ns]


def get_window_starts_s(run: ClockDriftRun) -> list[float]:
    return [window.start_ns / NS_PER_S for window in run.windows]


def get_window_lows_us(run: ClockDriftRun) -> list[float]:
    return [window.lowest_phase_ns / NS_PER_US for window in run.windows]


def plot_offset(run: ClockDriftRun) -> Figure:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    phases_us = [phase / NS_PER_US for phase in run.phases_ns]
    axis.plot(get_times_s(run), phases_us, '.', color=MUTED, markersize=1.5, label='carimbo')
    axis.step(
        get_window_starts_s(run),
        get_window_lows_us(run),
        where='post',
        color=SERIES,
        linewidth=2,
        label='mínimo da janela',
    )
    axis.set_title(f'fase dos carimbos: deriva {run.measure_drift_ppm():+.2f} ppm', loc='left')
    axis.set_xlabel('tempo desde o primeiro carimbo (s)')
    axis.set_ylabel('fase (µs)')
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def get_window_changes_us(run: ClockDriftRun) -> list[float]:
    """Return the change of the low between consecutive windows, in us, at least the axis end."""
    lows_us = get_window_lows_us(run)
    return [
        max(abs(after - before), SMALLEST_CHANGE_US)
        for before, after in itertools.pairwise(lows_us)
    ]


def plot_window_changes(run: ClockDriftRun) -> Figure:
    changes_us = get_window_changes_us(run)
    edges = np.geomspace(SMALLEST_CHANGE_US, max(changes_us) * 1.5, HISTOGRAM_BINS)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.hist(changes_us, bins=edges, color=SERIES, edgecolor=SURFACE, linewidth=0.5)
    threshold_us = JUMP_THRESHOLD_NS / NS_PER_US
    axis.axvline(threshold_us, color=MUTED, linestyle='--', linewidth=1)
    axis.annotate(
        f'limiar dos saltos {threshold_us:.0f} µs',
        (threshold_us, 1),
        xytext=(4, 4),
        textcoords='offset points',
        xycoords=('data', 'axes fraction'),
        va='top',
        color=MUTED,
    )
    axis.set_xscale('log')
    axis.set_yscale('log')
    axis.set_title('variação do mínimo da fase entre janelas consecutivas', loc='left')
    axis.set_xlabel('|variação| (µs)')
    axis.set_ylabel('pares de janelas')
    figure.tight_layout()
    return figure


def plot_jumps(run: ClockDriftRun) -> Figure:
    starts_s = get_window_starts_s(run)
    jumps = [
        (starts_s[position], jump_ns / NS_PER_US)
        for position, jump_ns in zip(run.find_jump_positions(), run.get_jumps_ns(), strict=True)
    ]
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.plot([t for t, _ in jumps], [r for _, r in jumps], 'o', color=SERIES, markersize=4)
    if len(jumps) >= 2:
        mean_us = statistics.mean(r for _, r in jumps)
        spread_us = statistics.stdev(r for _, r in jumps)
        axis.axhline(mean_us, color=MUTED, linestyle='--', linewidth=1)
        axis.set_title(
            f'saltos nas ressincronizações: média {mean_us:+.1f} µs, desvio {spread_us:.1f} µs',
            loc='left',
        )
    axis.set_xlabel('tempo desde o primeiro carimbo (s)')
    axis.set_ylabel('salto do offset (µs)')
    figure.tight_layout()
    return figure


def main() -> None:
    if len(sys.argv) != 3:
        print('usage: plot_clock_drift <bag directory> <output prefix>')
        sys.exit(2)
    run = ClockDriftRun(ControllerRecording.from_bag(sys.argv[1]))
    prefix = sys.argv[2]
    plot_offset(run).savefig(f'{prefix}_offset.png', dpi=DPI)
    plot_window_changes(run).savefig(f'{prefix}_janelas.png', dpi=DPI)
    plot_jumps(run).savefig(f'{prefix}_saltos.png', dpi=DPI)


if __name__ == '__main__':
    main()
