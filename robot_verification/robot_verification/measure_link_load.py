"""
Bench measurement of the link load (ADR 025): bytes per second in each direction of the UART.

Criticality: T1, evidence (robot-docs codigo.md).

The kernel counts the bytes of each serial port; the difference over DURATION_S gives the rate,
reported as a fraction of the UART capacity. The UART is full duplex, so the 50 % budget applies
to each direction. Start the agent and the load (e.g. the command at 20 Hz) before running this.
Reading the counters needs root: the script calls `sudo -n cat`.
"""

from pathlib import Path
import re
import subprocess
import sys
import time

DEVICE = 'ttyAMA0'  # RPi GPIO 14/15 (ADR 043)
COUNTERS_FILE = '/proc/tty/driver/ttyAMA'
BAUD = 921600
BITS_PER_BYTE = 10  # 8N1: start bit, 8 data bits, stop bit
CAPACITY_BYTES_PER_S = BAUD / BITS_PER_BYTE
BUDGET = 0.5  # ADR 025
DURATION_S = 60.0
# Kernel direction -> link direction.
DIRECTIONS = {'tx': 'RPi -> ESP32', 'rx': 'ESP32 -> RPi'}


def port_index():
    return Path(f'/sys/class/tty/{DEVICE}/line').read_text().strip()


def read_counters(index):
    """Return the bytes transmitted and received by the port since boot, by kernel direction."""
    text = subprocess.run(
        ['sudo', '-n', 'cat', COUNTERS_FILE], capture_output=True, text=True, check=True
    ).stdout
    line = next(line for line in text.splitlines() if line.startswith(f'{index}:'))
    return {
        direction: int(re.search(rf'\b{direction}:(\d+)', line).group(1))
        for direction in DIRECTIONS
    }


def main():
    index = port_index()
    before = read_counters(index)
    time.sleep(DURATION_S)
    after = read_counters(index)

    over_budget = False
    for direction, label in DIRECTIONS.items():
        rate = (after[direction] - before[direction]) / DURATION_S
        load = rate / CAPACITY_BYTES_PER_S
        over_budget |= load > BUDGET
        print(
            f'{label}: {rate:.0f} B/s = {load:.1%} of {CAPACITY_BYTES_PER_S:.0f} B/s '
            f'(budget {BUDGET:.0%})'
        )
    sys.exit(1 if over_budget else 0)


if __name__ == '__main__':
    main()
