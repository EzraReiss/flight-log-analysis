"""Per-ESC efficiency-response analysis by voltage band."""

import matplotlib.pyplot as plt
import numpy as np

from .. import runtime
from ..constants import COLORS
from .common import setup_style

def plot_efficiency_power_voltage_curves(
        esc_data, derived, active_escs, title_prefix, save_path, mode='pct'):
    """Show a readable f(power) using binned medians split by voltage band.

    The line is the median response and the translucent band is the middle 50%
    of samples in that power/voltage bin. Median elapsed time in each voltage
    band is included in the legend to expose voltage/time confounding.
    """
    setup_style()

    ylabel = 'Apparent Efficiency (%)' if mode == 'pct' else 'Efficiency (RPM/W)'
    data_key = 'efficiency_pct' if mode == 'pct' else 'efficiency_rpm_w'
    selected = [
        i for i in active_escs
        if i in esc_data and i in derived.get('per_esc', {})
    ]
    rows_by_esc = {}
    all_voltages, all_powers, all_efficiencies = [], [], []

    for i in selected:
        data = esc_data[i]
        deriv = derived['per_esc'][i]
        times = np.asarray(data.get('time', []), dtype=float)
        volts = np.asarray(deriv.get('volt_filtered', data.get('volt', [])), dtype=float)
        currents = np.asarray(deriv.get('curr_filtered', data.get('curr', [])), dtype=float)
        powers = np.asarray(deriv.get('power', []), dtype=float)
        efficiencies = np.asarray(
            deriv.get(data_key, deriv.get('efficiency', [])), dtype=float
        )
        n = min(len(times), len(volts), len(currents), len(powers), len(efficiencies))
        times, volts = times[:n], volts[:n]
        currents, powers = currents[:n], powers[:n]
        efficiencies = efficiencies[:n]
        valid = (
            np.isfinite(times) & np.isfinite(volts) & np.isfinite(currents)
            & np.isfinite(powers) & np.isfinite(efficiencies)
            & (currents >= runtime.options.min_current_threshold) & (powers > 0)
            & (efficiencies > 0)
        )
        if mode == 'pct':
            valid &= efficiencies <= 130
        rows = {
            'time': times[valid], 'volt': volts[valid],
            'power': powers[valid], 'efficiency': efficiencies[valid],
        }
        rows_by_esc[i] = rows
        all_voltages.extend(rows['volt'].tolist())
        all_powers.extend(rows['power'].tolist())
        all_efficiencies.extend(rows['efficiency'].tolist())

    if not selected or not all_voltages or not all_powers:
        print('No valid data available for the efficiency response-curve plot.')
        return

    voltage_min, voltage_max = np.percentile(all_voltages, [0.5, 99.5])
    if voltage_max <= voltage_min:
        voltage_max = voltage_min + 1.0
    voltage_edges = np.linspace(voltage_min, voltage_max, 5)
    power_min, power_max = np.percentile(all_powers, [1, 99])
    if power_max <= power_min:
        power_max = power_min + 1.0
    power_edges = np.linspace(power_min, power_max, 29)

    column_count = 2 if len(selected) > 1 else 1
    row_count = int(np.ceil(len(selected) / column_count))
    fig, axes = plt.subplots(
        row_count, column_count, figsize=(14, max(5.5, 5.0 * row_count)),
        sharex=True, sharey=True, squeeze=False
    )
    fig.suptitle(
        f'{title_prefix} - {ylabel} vs ESC Input Power by Voltage Band '
        f'[Current >= {runtime.options.min_current_threshold:g}A]', fontsize=14
    )
    color_map = plt.get_cmap('viridis')

    for panel, i in enumerate(selected):
        ax = axes.flat[panel]
        rows = rows_by_esc[i]
        plotted = 0
        for band_index in range(len(voltage_edges) - 1):
            lower, upper = voltage_edges[band_index:band_index + 2]
            if band_index == len(voltage_edges) - 2:
                in_voltage_band = (rows['volt'] >= lower) & (rows['volt'] <= upper)
            else:
                in_voltage_band = (rows['volt'] >= lower) & (rows['volt'] < upper)
            band_count = np.count_nonzero(in_voltage_band)
            if band_count < 20:
                continue

            median_time = float(np.median(rows['time'][in_voltage_band]))
            curve_power, curve_median, curve_q1, curve_q3 = [], [], [], []
            for power_start, power_end in zip(power_edges[:-1], power_edges[1:]):
                in_bin = (
                    in_voltage_band & (rows['power'] >= power_start)
                    & (rows['power'] < power_end)
                )
                if np.count_nonzero(in_bin) < 20:
                    continue
                values = rows['efficiency'][in_bin]
                curve_power.append(float(np.median(rows['power'][in_bin])))
                curve_median.append(float(np.median(values)))
                curve_q1.append(float(np.percentile(values, 25)))
                curve_q3.append(float(np.percentile(values, 75)))

            if len(curve_power) < 2:
                continue
            band_voltage = 0.5 * (lower + upper)
            normalized_voltage = (band_voltage - voltage_min) / (voltage_max - voltage_min)
            color = color_map(normalized_voltage)
            label = f'{lower:.1f}-{upper:.1f} V (median t={median_time:.0f}s)'
            ax.fill_between(curve_power, curve_q1, curve_q3, color=color, alpha=0.12)
            ax.plot(
                curve_power, curve_median, color=color, linewidth=2.2,
                marker='o', markersize=3.2, label=label
            )
            plotted += 1

        ax.set_title(f'ESC {i}')
        ax.set_xlabel('ESC Input Power (W = V x I)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(power_min, power_max)
        if mode == 'pct':
            visible_eff = np.asarray(all_efficiencies)
            y_low = max(0.0, float(np.percentile(visible_eff, 1)) - 5.0)
            y_high = min(110.0, float(np.percentile(visible_eff, 99)) + 5.0)
            ax.set_ylim(y_low, max(y_high, y_low + 10.0))
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(loc='best', fontsize=8, title='ESC voltage; median elapsed time')
        else:
            ax.text(0.5, 0.5, 'Not enough samples', ha='center', va='center', transform=ax.transAxes)

    for empty_panel in range(len(selected), axes.size):
        axes.flat[empty_panel].set_visible(False)

    fig.text(
        0.5, 0.012,
        'Line = median in each power bin; shaded region = middle 50% of samples. '
        'Voltage and elapsed time are strongly linked in this discharge test.',
        ha='center', fontsize=9
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
    fig.savefig(save_path, dpi=150)
    print(f'Saved efficiency response curves: {save_path}')
    plt.show()
