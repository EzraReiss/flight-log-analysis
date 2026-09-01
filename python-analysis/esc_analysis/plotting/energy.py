"""Energy integration and interactive voltage-versus-time selection."""

import matplotlib.pyplot as plt
import numpy as np

from ..constants import COLORS
from ..metrics import build_synchronized_total_series
from .common import setup_style

def _integrate_finite_pairs(times, values):
    """Trapezoidal integral without bridging NaN/dropout intervals."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(t) < 2 or len(y) < 2:
        return 0.0
    n = min(len(t), len(y))
    t, y = t[:n], y[:n]
    dt = np.diff(t)
    valid_pairs = np.isfinite(y[:-1]) & np.isfinite(y[1:]) & (dt > 0)
    areas = np.where(valid_pairs, ((y[:-1] + y[1:]) / 2) * dt, 0.0)
    return float(np.sum(areas))


def calculate_energy_between_times(esc_data, per_esc, selected_escs, start_s, end_s):
    """Integrate average ESC voltage × summed ESC current over a time interval."""
    synchronized = build_synchronized_total_series(esc_data, per_esc, selected_escs)
    if not synchronized['time']:
        return None
    t = np.asarray(synchronized['time'], dtype=float)
    interval = (t >= start_s) & (t <= end_s)
    if np.count_nonzero(interval) < 2:
        return None
    ti = t[interval]
    current = np.asarray(synchronized['curr'], dtype=float)[interval]
    voltage = np.asarray(synchronized['avg_volt'], dtype=float)[interval]
    power_avg_voltage = np.asarray(synchronized['power'], dtype=float)[interval]
    power_sum_esc = np.asarray(synchronized['sum_esc_power'], dtype=float)[interval]
    counts = np.asarray(synchronized['esc_count'], dtype=int)[interval]

    energy_avg_v_wh = _integrate_finite_pairs(ti, power_avg_voltage) / 3600
    energy_sum_esc_wh = _integrate_finite_pairs(ti, power_sum_esc) / 3600
    charge_ah = _integrate_finite_pairs(ti, current) / 3600
    voltage_time = _integrate_finite_pairs(ti, np.where(np.isfinite(voltage), voltage, np.nan))
    valid_voltage_seconds = _integrate_finite_pairs(ti, np.where(np.isfinite(voltage), 1.0, np.nan))
    mean_voltage = voltage_time / valid_voltage_seconds if valid_voltage_seconds > 0 else float('nan')
    valid_counts = counts[counts > 0]
    grid_dt = float(np.median(np.diff(ti))) if len(ti) >= 2 else 0.0
    reporting_seconds = {
        int(count): float(np.count_nonzero(counts == count) * grid_dt)
        for count in sorted(set(counts.tolist()))
    }

    return {
        'start_s': float(start_s),
        'end_s': float(end_s),
        'duration_s': float(end_s - start_s),
        'energy_wh': energy_avg_v_wh,
        'sum_esc_energy_wh': energy_sum_esc_wh,
        'method_difference_wh': energy_avg_v_wh - energy_sum_esc_wh,
        'charge_ah': charge_ah,
        'mean_voltage': mean_voltage,
        'min_reporting_escs': int(np.min(valid_counts)) if len(valid_counts) else 0,
        'max_reporting_escs': int(np.max(valid_counts)) if len(valid_counts) else 0,
        'reporting_seconds': reporting_seconds,
    }


def plot_interactive_voltage_energy_curve(esc_data, derived, active_escs, title_prefix, save_path):
    """Show voltage/current versus time and measure energy between clicked times."""
    synchronized = build_synchronized_total_series(
        esc_data, derived.get('per_esc', {}), active_escs
    )
    if not synchronized['time']:
        print("No synchronized ESC data available for the voltage-time curve.")
        return

    time_s = np.asarray(synchronized['time'], dtype=float)
    voltage = np.asarray(synchronized['avg_volt'], dtype=float)
    current = np.asarray(synchronized['curr'], dtype=float)
    power = np.asarray(synchronized['power'], dtype=float)
    esc_counts = np.asarray(synchronized['esc_count'], dtype=int)

    energy_steps = np.zeros(len(time_s), dtype=float)
    charge_steps = np.zeros(len(time_s), dtype=float)
    if len(time_s) >= 2:
        dt = np.diff(time_s)
        valid_power_pairs = (
            np.isfinite(power[:-1]) & np.isfinite(power[1:]) & (dt > 0)
        )
        valid_current_pairs = (
            np.isfinite(current[:-1]) & np.isfinite(current[1:]) & (dt > 0)
        )
        energy_steps[1:] = np.where(
            valid_power_pairs,
            ((power[:-1] + power[1:]) / 2) * dt / 3600,
            0.0
        )
        charge_steps[1:] = np.where(
            valid_current_pairs,
            ((current[:-1] + current[1:]) / 2) * dt / 3600,
            0.0
        )
    cumulative_wh = np.cumsum(energy_steps)
    cumulative_ah = np.cumsum(charge_steps)

    valid = np.isfinite(voltage) & np.isfinite(cumulative_wh) & (esc_counts > 0)
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) < 2:
        print("Not enough valid voltage data to draw the curve.")
        return

    setup_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(
        f"{title_prefix} - Voltage and Summed ESC Current / Energy Interval",
        fontsize=14
    )
    curve_time = np.where(valid, time_s, np.nan)
    curve_voltage = np.where(valid, voltage, np.nan)
    curve_current = np.where(valid, current, np.nan)
    voltage_line, = ax.plot(
        curve_time, curve_voltage, color=COLORS[0], linewidth=1.6,
        label='Average ESC voltage'
    )

    current_axis = ax.twinx()
    current_line, = current_axis.plot(
        curve_time, curve_current, color=COLORS[3], linewidth=1.4,
        alpha=0.85, label='Summed ESC current'
    )

    # Keep the scatter responsive on large logs while retaining the full curve
    # for nearest-point selection and energy calculations.
    plot_step = max(1, len(valid_indices) // 3500)
    plotted = valid_indices[::plot_step]
    ax.scatter(
        time_s[plotted], voltage[plotted], color=COLORS[0],
        s=8, alpha=0.35, edgecolors='none'
    )
    ax.set_xlabel('Time since selected log start (s)')
    ax.set_ylabel('Average reporting-ESC voltage (V)', color=COLORS[0])
    ax.tick_params(axis='y', labelcolor=COLORS[0])
    current_axis.set_ylabel('Summed current across reporting ESCs (A)', color=COLORS[3])
    current_axis.tick_params(axis='y', labelcolor=COLORS[3])
    ax.legend(
        [voltage_line, current_line],
        [voltage_line.get_label(), current_line.get_label()],
        loc='lower left'
    )
    ax.grid(True, alpha=0.3)

    instruction = ax.text(
        0.01, 0.99,
        'Left-click two times. Right-click to clear.',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )
    result_text = ax.text(
        0.01, 0.91, 'No points selected', transform=ax.transAxes,
        va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    selected_indices = []
    selection_artists = []

    def clear_selection():
        selected_indices.clear()
        while selection_artists:
            artist = selection_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        result_text.set_text('No points selected')
        fig.canvas.draw_idle()

    def nearest_curve_index(event):
        return int(valid_indices[int(np.argmin(np.abs(time_s[valid_indices] - event.xdata)))])

    def draw_selection():
        while selection_artists:
            artist = selection_artists.pop()
            try:
                artist.remove()
            except Exception:
                pass
        for number, idx in enumerate(selected_indices, start=1):
            marker = ax.scatter(
                [time_s[idx]], [voltage[idx]], s=85,
                facecolors='none', edgecolors=COLORS[(number - 1) % len(COLORS)],
                linewidths=2.2, zorder=8
            )
            label = ax.annotate(
                f"P{number}: {time_s[idx]:.1f}s, {voltage[idx]:.3f}V",
                (time_s[idx], voltage[idx]), xytext=(8, 8),
                textcoords='offset points', fontsize=9,
                color=COLORS[(number - 1) % len(COLORS)]
            )
            selection_artists.extend([marker, label])

        if len(selected_indices) == 1:
            idx = selected_indices[0]
            result_text.set_text(
                f"P1  {voltage[idx]:.3f} V | {current[idx]:.2f} A | "
                f"{cumulative_wh[idx]:.2f} Wh | {cumulative_ah[idx]:.3f} Ah | "
                f"t={time_s[idx]:.1f}s | ESCs={esc_counts[idx]}"
            )
        elif len(selected_indices) == 2:
            first, second = sorted(selected_indices, key=lambda idx: time_s[idx])
            delta_v = voltage[second] - voltage[first]
            energy_used = cumulative_wh[second] - cumulative_wh[first]
            charge_used = cumulative_ah[second] - cumulative_ah[first]
            duration_s = time_s[second] - time_s[first]
            mean_current = charge_used * 3600 / duration_s if duration_s > 0 else float('nan')
            mean_power = energy_used * 3600 / duration_s if duration_s > 0 else float('nan')
            segment = slice(first, second + 1)
            highlight, = ax.plot(
                curve_time[segment], curve_voltage[segment],
                color=COLORS[2 % len(COLORS)], linewidth=2.8,
                alpha=0.9, zorder=6
            )
            current_highlight, = current_axis.plot(
                curve_time[segment], curve_current[segment],
                color=COLORS[1 % len(COLORS)], linewidth=2.6,
                alpha=0.9, zorder=6
            )
            selection_artists.extend([highlight, current_highlight])
            result_text.set_text(
                f"V1={voltage[first]:.3f} V   V2={voltage[second]:.3f} V   "
                f"ΔV={delta_v:+.3f} V\n"
                f"Energy used={energy_used:.3f} Wh   Charge used={charge_used:.3f} Ah   "
                f"Δt={duration_s:.1f}s\n"
                f"Mean current={mean_current:.2f} A   Mean power={mean_power:.1f} W"
            )
            fig.savefig(save_path, dpi=140)
            print(
                f"Selected {voltage[first]:.3f} V -> {voltage[second]:.3f} V: "
                f"{energy_used:.3f} Wh, {charge_used:.3f} Ah "
                f"(delta V {delta_v:+.3f} V)"
            )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes not in (ax, current_axis):
            return
        if event.button == 3:
            clear_selection()
            return
        if event.button != 1:
            return
        if len(selected_indices) >= 2:
            clear_selection()
        selected_indices.append(nearest_curve_index(event))
        draw_selection()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout()
    fig.savefig(save_path, dpi=140)
    print(f"Saved interactive voltage-time image: {save_path}")
    print("Click two times in the graph window; right-click to reset.")
    plt.show()
