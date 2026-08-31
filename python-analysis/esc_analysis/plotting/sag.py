"""Throttle-step voltage-sag and resistance-event analysis."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import runtime
from ..config import get_current_scale
from ..constants import COLORS
from ..telemetry import _interpolate_without_bridging_gaps
from .common import setup_style

def _detect_throttle_drop_resistance_events(
        esc_data, active_escs, min_throttle_drop_pwm=60.0,
        max_reliable_rpm=6000.0, current_scale_rules=None):
    """Measure local system resistance from stable throttle step-down pairs.

    Raw (unfiltered) ESC telemetry is synchronized so an adjustable display
    filter cannot smear the voltage/current step. Each event compares robust
    medians on plateaus before and after the transition.
    """
    selected = [
        i for i in sorted(active_escs)
        if i in esc_data and len(esc_data[i].get('time', [])) >= 2
    ]
    if not selected:
        return {}, []

    time_arrays = [np.asarray(esc_data[i]['time'], dtype=float) for i in selected]
    positive_steps = np.concatenate([
        np.diff(values)[np.diff(values) > 0]
        for values in time_arrays if len(values) >= 2
    ])
    grid_dt = float(np.median(positive_steps)) if len(positive_steps) else 0.25
    grid_dt = min(1.0, max(0.05, grid_dt))
    grid = np.arange(
        min(float(values[0]) for values in time_arrays),
        max(float(values[-1]) for values in time_arrays) + grid_dt * 0.5,
        grid_dt
    )

    voltage_rows, current_rows = [], []
    rpm_rows, throttle_rows, temp_rows = [], [], []
    for i in selected:
        data = esc_data[i]
        sample_throttle = np.asarray(data.get('throttle', []), dtype=float)
        sample_current = np.asarray(data.get('curr', []), dtype=float)
        n = min(len(sample_current), len(sample_throttle))
        if current_scale_rules and n:
            scales = np.asarray([
                get_current_scale(value, current_scale_rules)
                for value in sample_throttle[:n]
            ])
            scaled_current = sample_current.copy()
            scaled_current[:n] *= scales
        else:
            scaled_current = sample_current

        voltage_rows.append(_interpolate_without_bridging_gaps(
            data['time'], data.get('volt', []), grid
        ))
        current_rows.append(_interpolate_without_bridging_gaps(
            data['time'], scaled_current, grid
        ))
        rpm_rows.append(_interpolate_without_bridging_gaps(
            data['time'], data.get('rpm', []), grid
        ))
        throttle_rows.append(_interpolate_without_bridging_gaps(
            data['time'], data.get('throttle', []), grid
        ))
        temp_rows.append(_interpolate_without_bridging_gaps(
            data['time'], data.get('temp', []), grid
        ))

    voltage_matrix = np.vstack(voltage_rows)
    current_matrix = np.vstack(current_rows)
    rpm_matrix = np.vstack(rpm_rows)
    throttle_matrix = np.vstack(throttle_rows)
    temp_matrix = np.vstack(temp_rows)
    reporting = np.isfinite(voltage_matrix) & np.isfinite(current_matrix)
    reporting_count = np.sum(reporting, axis=0)
    expected_count = int(np.max(reporting_count)) if len(reporting_count) else 0

    voltage_sum = np.nansum(np.where(reporting, voltage_matrix, np.nan), axis=0)
    system_voltage = np.divide(
        voltage_sum, reporting_count,
        out=np.full(len(grid), np.nan), where=reporting_count > 0
    )
    system_current = np.nansum(np.where(reporting, current_matrix, np.nan), axis=0)
    system_current[reporting_count == 0] = np.nan
    system_power = np.nansum(
        np.where(reporting, voltage_matrix * current_matrix, np.nan), axis=0
    )
    system_power[reporting_count == 0] = np.nan

    throttle_valid = np.any(np.isfinite(throttle_matrix), axis=0)
    system_throttle = np.full(len(grid), np.nan)
    if np.any(throttle_valid):
        system_throttle[throttle_valid] = np.nanmax(
            throttle_matrix[:, throttle_valid], axis=0
        )
    temp_valid = reporting & np.isfinite(temp_matrix)
    temp_count = np.sum(temp_valid, axis=0)
    system_temp = np.divide(
        np.nansum(np.where(temp_valid, temp_matrix, np.nan), axis=0), temp_count,
        out=np.full(len(grid), np.nan), where=temp_count > 0
    )

    # Integrate only periods with the complete selected ESC set. This avoids
    # pretending that the early two-ESC telemetry segment measured all four.
    cumulative_wh = np.full(len(grid), np.nan)
    full_indices = np.flatnonzero(reporting_count == expected_count)
    if len(full_indices):
        first_full = int(full_indices[0])
        cumulative_wh[first_full] = 0.0
        running_wh = 0.0
        for j in range(first_full + 1, len(grid)):
            if (
                    reporting_count[j - 1] == expected_count
                    and reporting_count[j] == expected_count
                    and np.isfinite(system_power[j - 1])
                    and np.isfinite(system_power[j])
                    and 0 < grid[j] - grid[j - 1] <= 2.5 * grid_dt):
                running_wh += (
                    max(0.0, system_power[j - 1]) + max(0.0, system_power[j])
                ) * 0.5 * (grid[j] - grid[j - 1]) / 3600.0
            cumulative_wh[j] = running_wh

    if not np.any(np.isfinite(system_throttle)):
        return {
            'time': grid, 'voltage': system_voltage, 'current': system_current,
            'energy_wh': cumulative_wh, 'expected_count': expected_count,
        }, []

    throttle_window = max(3, int(round(0.5 / grid_dt)))
    smoothed_throttle = pd.Series(system_throttle).rolling(
        throttle_window, center=True, min_periods=1
    ).median().to_numpy()
    throttle_change = np.diff(smoothed_throttle)
    trigger_drop = max(8.0, 0.15 * min_throttle_drop_pwm)
    candidate_indices = np.flatnonzero(
        np.isfinite(throttle_change) & (throttle_change <= -trigger_drop)
    ) + 1

    groups = []
    for index in candidate_indices:
        if not groups or grid[index] - grid[groups[-1][-1]] > 1.0:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))

    finite_throttle = system_throttle[np.isfinite(system_throttle)]
    idle_throttle_limit = float(np.percentile(finite_throttle, 5)) + 50.0
    minimum_current_step = max(
        15.0, 0.5 * runtime.options.min_current_threshold * max(expected_count, 1)
    )
    events = []
    for event_number, group in enumerate(groups, start=1):
        start_index, end_index = group[0], group[-1]
        transition_start = grid[start_index]
        transition_end = grid[end_index]
        pre_mask = (
            (grid >= transition_start - 1.8)
            & (grid <= transition_start - 0.4)
            & (reporting_count == expected_count)
        )
        post_mask = (
            (grid >= transition_end + 0.5)
            & (grid <= transition_end + 1.9)
            & (reporting_count == expected_count)
        )
        event_time = 0.5 * (transition_start + transition_end)
        event_index = int(np.argmin(np.abs(grid - event_time)))
        event = {
            'event_id': event_number,
            'time_s': event_time,
            'energy_since_full_telemetry_wh': (
                float(cumulative_wh[event_index])
                if np.isfinite(cumulative_wh[event_index]) else np.nan
            ),
            'accepted': False,
            'rejection_reason': '',
        }
        reasons = []
        if np.count_nonzero(pre_mask) < 4 or np.count_nonzero(post_mask) < 4:
            reasons.append('short or incomplete plateau')
            event['rejection_reason'] = '; '.join(reasons)
            events.append(event)
            continue

        pre_current_each = np.nanmedian(current_matrix[:, pre_mask], axis=1)
        post_current_each = np.nanmedian(current_matrix[:, post_mask], axis=1)
        pre_rpm_each = np.nanmedian(rpm_matrix[:, pre_mask], axis=1)
        pre_throttle = float(np.nanmedian(system_throttle[pre_mask]))
        post_throttle = float(np.nanmedian(system_throttle[post_mask]))
        pre_current = float(np.nansum(pre_current_each))
        post_current = float(np.nansum(post_current_each))
        pre_voltage = float(np.nanmedian(system_voltage[pre_mask]))
        post_voltage = float(np.nanmedian(system_voltage[post_mask]))
        delta_current = pre_current - post_current
        delta_voltage = post_voltage - pre_voltage
        resistance_mohm = (
            1000.0 * delta_voltage / delta_current
            if delta_current > 0 else np.nan
        )
        mean_voltage = 0.5 * (pre_voltage + post_voltage)
        pre_rpm = float(np.nanmax(pre_rpm_each))
        mean_temp = float(np.nanmedian(np.concatenate([
            system_temp[pre_mask], system_temp[post_mask]
        ])))
        throttle_drop = pre_throttle - post_throttle
        throttle_spread = max(
            float(np.nanpercentile(system_throttle[pre_mask], 90)
                  - np.nanpercentile(system_throttle[pre_mask], 10)),
            float(np.nanpercentile(system_throttle[post_mask], 90)
                  - np.nanpercentile(system_throttle[post_mask], 10))
        )
        pre_current_iqr = float(
            np.nanpercentile(system_current[pre_mask], 75)
            - np.nanpercentile(system_current[pre_mask], 25)
        )
        post_is_idle = post_throttle <= idle_throttle_limit

        if throttle_drop < min_throttle_drop_pwm:
            reasons.append('throttle drop too small')
        if throttle_spread > 35.0:
            reasons.append('throttle plateau not stable')
        if np.any(~np.isfinite(pre_current_each)) or np.any(
                pre_current_each < runtime.options.min_current_threshold):
            reasons.append('pre-step ESC current below reliable range')
        if not post_is_idle and (
                np.any(~np.isfinite(post_current_each))
                or np.any(post_current_each < runtime.options.min_current_threshold)):
            reasons.append('post-step ESC current below reliable range')
        if delta_current < minimum_current_step:
            reasons.append('current step too small')
        if pre_current_iqr > max(8.0, 0.15 * pre_current):
            reasons.append('pre-step current not stable')
        if max_reliable_rpm is not None and pre_rpm > max_reliable_rpm:
            reasons.append('RPM above current-telemetry limit')
        if not np.isfinite(delta_voltage) or delta_voltage <= 0.05:
            reasons.append('voltage did not recover clearly')
        if not np.isfinite(resistance_mohm) or not 1.0 <= resistance_mohm <= 100.0:
            reasons.append('implausible resistance')

        event.update({
            'event_type': 'idle return' if post_is_idle else 'partial step',
            'pre_throttle_pwm': pre_throttle,
            'post_throttle_pwm': post_throttle,
            'throttle_drop_pwm': throttle_drop,
            'pre_current_a': pre_current,
            'post_current_a': post_current,
            'delta_current_a': delta_current,
            'pre_voltage_v': pre_voltage,
            'post_voltage_v': post_voltage,
            'delta_voltage_v': delta_voltage,
            'mean_voltage_v': mean_voltage,
            'resistance_mohm': resistance_mohm,
            'pre_rpm': pre_rpm,
            'mean_esc_temp_c': mean_temp,
            'pre_current_iqr_a': pre_current_iqr,
            'accepted': not reasons,
            'rejection_reason': '; '.join(reasons),
        })
        events.append(event)

    series = {
        'time': grid, 'voltage': system_voltage, 'current': system_current,
        'energy_wh': cumulative_wh, 'expected_count': expected_count,
        'minimum_current_step': minimum_current_step,
        'idle_throttle_limit': idle_throttle_limit,
    }
    return series, events


def plot_voltage_sag(
        esc_data, derived, active_escs, title_prefix, save_path, esc_count=4,
        esc_count_series=None, min_throttle_drop_pwm=60.0,
        max_reliable_rpm=6000.0, current_scale_rules=None):
    """Estimate local battery/system resistance from throttle step-down events."""
    del derived, esc_count, esc_count_series  # Kept in the signature for compatibility.
    setup_style()
    series, events = _detect_throttle_drop_resistance_events(
        esc_data,
        active_escs,
        min_throttle_drop_pwm=min_throttle_drop_pwm,
        max_reliable_rpm=max_reliable_rpm,
        current_scale_rules=current_scale_rules,
    )
    accepted = [event for event in events if event.get('accepted')]
    event_csv_path = os.path.splitext(save_path)[0] + '_events.csv'
    if events:
        pd.DataFrame(events).to_csv(event_csv_path, index=False)
        print(f'Saved throttle-step event table: {event_csv_path}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    median_resistance = (
        float(np.median([event['resistance_mohm'] for event in accepted]))
        if accepted else np.nan
    )
    summary = (
        f'{len(accepted)}/{len(events)} accepted throttle drops'
        + (f' | Median R = {median_resistance:.1f} mOhm' if accepted else '')
    )
    fig.suptitle(
        f'{title_prefix} - Throttle-Step System Resistance\n{summary}',
        fontsize=14
    )

    if not accepted:
        for ax in axes.flat:
            ax.text(
                0.5, 0.5,
                'No valid throttle-drop events\n'
                f'(minimum drop {min_throttle_drop_pwm:g} PWM)',
                ha='center', va='center', transform=ax.transAxes
            )
            ax.grid(True, alpha=0.25)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(save_path, dpi=140)
        print(f'Saved: {save_path}')
        plt.show()
        return

    resistance = np.asarray([event['resistance_mohm'] for event in accepted])
    mean_voltage = np.asarray([event['mean_voltage_v'] for event in accepted])
    energy_wh = np.asarray([
        event['energy_since_full_telemetry_wh'] for event in accepted
    ])
    delta_current = np.asarray([event['delta_current_a'] for event in accepted])
    delta_voltage = np.asarray([event['delta_voltage_v'] for event in accepted])
    event_time = np.asarray([event['time_s'] for event in accepted])
    marker_size = np.clip(24.0 + 0.9 * delta_current, 34.0, 105.0)

    ax = axes[0, 0]
    scatter = ax.scatter(
        mean_voltage, resistance, c=energy_wh, cmap='viridis',
        s=marker_size, alpha=0.85, edgecolors='none'
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Reported ESC energy since full telemetry (Wh)')
    ax.axhline(median_resistance, color='black', linestyle='--', linewidth=1.2,
               label=f'Median {median_resistance:.1f} mOhm')
    ax.set_xlabel('Mean pack voltage across the step (V)')
    ax.set_ylabel('System resistance, delta V / delta I (mOhm)')
    ax.set_title('Resistance by pack voltage')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    scatter = ax.scatter(
        energy_wh, resistance, c=mean_voltage, cmap='viridis',
        s=marker_size, alpha=0.85, edgecolors='none'
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Mean pack voltage (V)')
    if len(accepted) >= 4:
        order = np.argsort(energy_wh)
        trend = pd.Series(resistance[order]).rolling(
            3, center=True, min_periods=2
        ).median().to_numpy()
        ax.plot(energy_wh[order], trend, color='black', linewidth=1.5,
                label='3-event rolling median')
        ax.legend(loc='best')
    ax.set_xlabel('Reported ESC energy since full telemetry (Wh)')
    ax.set_ylabel('System resistance, delta V / delta I (mOhm)')
    ax.set_title('Resistance through the discharge')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    scatter = ax.scatter(
        delta_current, delta_voltage, c=mean_voltage, cmap='viridis',
        s=marker_size, alpha=0.85, edgecolors='none'
    )
    current_line = np.linspace(0.0, max(delta_current) * 1.08, 100)
    ax.plot(
        current_line, current_line * median_resistance / 1000.0,
        color='black', linestyle='--', linewidth=1.5,
        label=f'delta V = {median_resistance / 1000.0:.4f} x delta I'
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Current decrease, delta I (A)')
    ax.set_ylabel('Voltage recovery, delta V (V)')
    ax.set_title('Measured throttle-step pairs (color = mean voltage)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    time = np.asarray(series['time'])
    voltage = np.asarray(series['voltage'])
    stride = max(1, int(np.ceil(len(time) / 5000)))
    ax.plot(time[::stride], voltage[::stride], color=COLORS[0], linewidth=1.0,
            alpha=0.7, label='ESC-reported pack voltage')
    ax.scatter(event_time, mean_voltage, color=COLORS[2], s=45,
               label='Accepted step event', zorder=3)
    rejected = [
        event for event in events
        if not event.get('accepted') and np.isfinite(event.get('mean_voltage_v', np.nan))
    ]
    if rejected:
        ax.scatter(
            [event['time_s'] for event in rejected],
            [event['mean_voltage_v'] for event in rejected],
            color=COLORS[3], marker='x', s=35,
            label='Rejected candidate', zorder=3
        )
    ax.set_xlabel('Elapsed time (s)')
    ax.set_ylabel('Pack voltage at ESCs (V)')
    ax.set_title('Accepted events across the test')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)

    rpm_note = (
        f'; pre-step RPM <= {max_reliable_rpm:g}'
        if max_reliable_rpm is not None else ''
    )
    fig.text(
        0.5, 0.012,
        f'Accepted events require stable plateaus, delta throttle >= '
        f'{min_throttle_drop_pwm:g} PWM, delta current >= '
        f'{series["minimum_current_step"]:.1f} A, and reliable pre-step ESC current'
        f'{rpm_note}. Resistance includes the battery, connectors, and wiring up to the ESC telemetry points.',
        ha='center', fontsize=9
    )
    fig.tight_layout(rect=[0.01, 0.045, 0.99, 0.93])
    fig.savefig(save_path, dpi=140)
    print(f'Saved throttle-step voltage sag analysis: {save_path}')
    print(
        f'Accepted {len(accepted)} of {len(events)} candidates; '
        f'median system resistance {median_resistance:.2f} mOhm.'
    )
    plt.show()
