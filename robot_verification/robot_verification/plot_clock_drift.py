"""
Charts of the clock drift between the ESP32 and the RPi (ADR 027) from the bag of V-07.

Criticality: T2, product (robot-docs codigo.md).

The reconstructed offset over time, a staircase whose slope is the drift; the histogram of the
residuals, where the conversion noise and the jumps separate and JUMP_THRESHOLD_NS is chosen; the
jumps over time, whose spread is the error of the synchronization. The numbers come from
measure_clock_drift.

Usage: plot_clock_drift <bag directory> <output prefix>. Writes <prefix>_offset.png,
<prefix>_residuos.png and <prefix>_saltos.png.
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
SMALLEST_RESIDUAL_US = 0.1  # left end of the logarithmic axis; a zero residual is put here

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


def get_offsets_us(run: ClockDriftRun) -> list[float]:
    """Return the reconstructed offset at each stamp, from the first, in us."""
    offsets_ns = itertools.accumulate(run.residuals_ns, initial=0)
    return [offset / NS_PER_US for offset in offsets_ns]


def plot_offset(run: ClockDriftRun) -> Figure:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.step(get_times_s(run), get_offsets_us(run), where='post', color=SERIES, linewidth=2)
    axis.set_title(f'offset reconstruído: deriva {run.measure_drift_ppm():+.2f} ppm', loc='left')
    axis.set_xlabel('tempo desde o primeiro carimbo (s)')
    axis.set_ylabel('offset (µs)')
    figure.tight_layout()
    return figure


def plot_residuals(run: ClockDriftRun) -> Figure:
    residuals_us = [max(abs(r) / NS_PER_US, SMALLEST_RESIDUAL_US) for r in run.residuals_ns]
    edges = np.geomspace(SMALLEST_RESIDUAL_US, max(residuals_us) * 1.5, HISTOGRAM_BINS)
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    axis.hist(residuals_us, bins=edges, color=SERIES, edgecolor=SURFACE, linewidth=0.5)
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
    axis.set_title('|resíduo| de cada intervalo até o múltiplo de T_c', loc='left')
    axis.set_xlabel('|resíduo| (µs)')
    axis.set_ylabel('intervalos')
    figure.tight_layout()
    return figure


def plot_jumps(run: ClockDriftRun) -> Figure:
    times_s = get_times_s(run)
    jumps = [(times_s[i + 1], run.residuals_ns[i] / NS_PER_US) for i in run.find_jump_positions()]
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
    plot_residuals(run).savefig(f'{prefix}_residuos.png', dpi=DPI)
    plot_jumps(run).savefig(f'{prefix}_saltos.png', dpi=DPI)


if __name__ == '__main__':
    main()
