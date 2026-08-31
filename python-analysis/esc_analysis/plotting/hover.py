"""Sustained-hover detection, thrust balance, CG, weight, and moment estimates."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymavlink import mavutil

from ..constants import COLORS
from ..telemetry import _interpolate_without_bridging_gaps
from .common import setup_style

def _interpolate_static_motor_quantity(rpm, motor_spec, quantity_key):
    """Interpolate a published static-test quantity against RPM.

    Outside the table, use the static-propeller square-law only for a modest
    extrapolation. Hover-window validation normally keeps points inside the
    published range.
    """
    quantity = motor_spec.get(quantity_key)
    if not quantity:
        return None
    throttle_rows = sorted(set(motor_spec['data']) & set(quantity))
    if len(throttle_rows) < 2:
        return None
    reference_rpm = np.asarray(
        [motor_spec['data'][row][4] for row in throttle_rows], dtype=float
    )
    reference_values = np.asarray([quantity[row] for row in throttle_rows], dtype=float)
    values = np.asarray(rpm, dtype=float)
    result = np.interp(values, reference_rpm, reference_values)
    below = np.isfinite(values) & (values < reference_rpm[0]) & (values > 0)
    above = np.isfinite(values) & (values > reference_rpm[-1])
    result[below] = reference_values[0] * (values[below] / reference_rpm[0]) ** 2
    result[above] = reference_values[-1] * (values[above] / reference_rpm[-1]) ** 2
    result[~np.isfinite(values) | (values <= 0)] = np.nan
    return result


def _load_hover_flight_state(filepath):
    """Load EKF velocity/attitude, relative altitude, and barometer pressure."""
    mlog = mavutil.mavlink_connection(filepath)
    ekf_rows, position_rows, pressure_rows = [], [], []
    while True:
        msg = mlog.recv_match(type=['XKF1', 'POS', 'BARO'])
        if msg is None:
            break
        msg_type = msg.get_type()
        if msg_type == 'XKF1' and int(getattr(msg, 'C', 0)) == 0:
            ekf_rows.append((
                msg.TimeUS / 1e6, float(msg.VN), float(msg.VE), float(msg.VD),
                float(msg.Roll), float(msg.Pitch)
            ))
        elif msg_type == 'POS':
            position_rows.append((msg.TimeUS / 1e6, float(msg.RelHomeAlt)))
        elif msg_type == 'BARO' and int(getattr(msg, 'I', 0)) == 0:
            pressure_rows.append((msg.TimeUS / 1e6, float(msg.Press)))
    return {
        'ekf': np.asarray(ekf_rows, dtype=float),
        'position': np.asarray(position_rows, dtype=float),
        'pressure': np.asarray(pressure_rows, dtype=float),
    }


def _true_segments(mask, time_values, minimum_duration):
    """Return inclusive index pairs for sustained True regions."""
    segments = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if value and index == len(mask) - 1 else index - 1
            if time_values[end] - time_values[start] >= minimum_duration:
                segments.append((start, end))
            start = None
    return segments


def analyze_hover_balance(
        filepath, esc_data, active_escs, motor_spec, hover_config=None):
    """Detect sustained hover and estimate thrust, CG balance, and moments."""
    hover_config = hover_config or {}
    required_escs = [0, 1, 2, 3]
    if not all(i in esc_data and i in active_escs for i in required_escs):
        return {'error': 'Hover balance requires ESC instances 0, 1, 2, and 3.'}
    if not motor_spec.get('thrust_gf') or not motor_spec.get('torque_nm'):
        return {'error': f"No static thrust/torque table is available for {motor_spec['name']}."}

    minimum_duration = float(hover_config.get('min_duration_sec', 5.0))
    minimum_rpm = float(hover_config.get('min_rpm', 3000.0))
    max_vertical_speed = float(hover_config.get('max_vertical_speed_mps', 0.5))
    max_horizontal_speed = float(hover_config.get('max_horizontal_speed_mps', 2.0))
    minimum_altitude = float(hover_config.get('min_altitude_m', 1.0))
    max_tilt = float(hover_config.get('max_tilt_deg', 15.0))
    ambient_temperature_c = float(hover_config.get('air_temperature_c', 25.0))
    half_length_m = hover_config.get('motor_half_length_m')
    half_width_m = hover_config.get('motor_half_width_m')
    half_length_m = float(half_length_m) if half_length_m is not None else None
    half_width_m = float(half_width_m) if half_width_m is not None else None

    time_arrays = {
        i: np.asarray(esc_data[i]['time_us'], dtype=float) / 1e6
        for i in required_escs
    }
    positive_steps = np.concatenate([
        np.diff(values)[np.diff(values) > 0] for values in time_arrays.values()
    ])
    grid_dt = float(np.median(positive_steps)) if len(positive_steps) else 0.2
    grid_dt = min(0.5, max(0.05, grid_dt))
    grid_start = max(float(values[0]) for values in time_arrays.values())
    grid_end = min(float(values[-1]) for values in time_arrays.values())
    if grid_end <= grid_start:
        return {'error': 'The four ESC telemetry streams do not overlap.'}
    time_abs = np.arange(grid_start, grid_end + 0.5 * grid_dt, grid_dt)
    display_time = time_abs - time_abs[0]

    rpm_matrix = np.vstack([
        _interpolate_without_bridging_gaps(
            time_arrays[i], esc_data[i].get('rpm', []), time_abs
        ) for i in required_escs
    ])
    all_rpm_valid = np.all(np.isfinite(rpm_matrix), axis=0)

    state = _load_hover_flight_state(filepath)
    ekf = state['ekf']
    if ekf.ndim != 2 or len(ekf) < 2:
        return {'error': 'XKF1 velocity/attitude data is unavailable.'}
    north_velocity = _interpolate_without_bridging_gaps(ekf[:, 0], ekf[:, 1], time_abs)
    east_velocity = _interpolate_without_bridging_gaps(ekf[:, 0], ekf[:, 2], time_abs)
    down_velocity = _interpolate_without_bridging_gaps(ekf[:, 0], ekf[:, 3], time_abs)
    roll_deg = _interpolate_without_bridging_gaps(ekf[:, 0], ekf[:, 4], time_abs)
    pitch_deg = _interpolate_without_bridging_gaps(ekf[:, 0], ekf[:, 5], time_abs)

    position = state['position']
    if position.ndim == 2 and len(position) >= 2:
        relative_altitude = _interpolate_without_bridging_gaps(
            position[:, 0], position[:, 1], time_abs
        )
        altitude_valid = np.isfinite(relative_altitude) & (relative_altitude >= minimum_altitude)
    else:
        relative_altitude = np.full(len(time_abs), np.nan)
        altitude_valid = np.ones(len(time_abs), dtype=bool)

    smoothing_samples = max(3, int(round(1.0 / grid_dt)))
    def smooth(values):
        return pd.Series(values).rolling(
            smoothing_samples, center=True, min_periods=max(2, smoothing_samples // 2)
        ).median().to_numpy()

    north_velocity = smooth(north_velocity)
    east_velocity = smooth(east_velocity)
    down_velocity = smooth(down_velocity)
    roll_deg = smooth(roll_deg)
    pitch_deg = smooth(pitch_deg)
    relative_altitude = smooth(relative_altitude)
    horizontal_speed = np.hypot(north_velocity, east_velocity)

    hover_candidate = (
        all_rpm_valid
        & np.all(rpm_matrix >= minimum_rpm, axis=0)
        & np.isfinite(down_velocity)
        & (np.abs(down_velocity) <= max_vertical_speed)
        & np.isfinite(horizontal_speed)
        & (horizontal_speed <= max_horizontal_speed)
        & np.isfinite(roll_deg) & (np.abs(roll_deg) <= max_tilt)
        & np.isfinite(pitch_deg) & (np.abs(pitch_deg) <= max_tilt)
        & altitude_valid
    )
    # Remove isolated state-estimator dropouts without bridging a real climb.
    hover_mask = pd.Series(hover_candidate.astype(float)).rolling(
        smoothing_samples, center=True, min_periods=1
    ).mean().to_numpy() >= 0.7
    segments = _true_segments(hover_mask, display_time, minimum_duration)
    if not segments:
        return {
            'error': (
                f'No hover window lasted {minimum_duration:g}s with |vertical speed| <= '
                f'{max_vertical_speed:g}m/s and altitude >= {minimum_altitude:g}m.'
            )
        }

    pressure = state['pressure']
    if pressure.ndim == 2 and len(pressure) >= 2:
        pressure_pa = _interpolate_without_bridging_gaps(
            pressure[:, 0], pressure[:, 1], time_abs
        )
    else:
        pressure_pa = np.full(len(time_abs), 101325.0)
    reference_temperature_k = 25.0 + 273.15
    actual_temperature_k = ambient_temperature_c + 273.15
    density_ratio = (
        pressure_pa / 101325.0 * reference_temperature_k / actual_temperature_k
    )
    density_ratio[~np.isfinite(density_ratio)] = 1.0

    thrust_gf_reference = np.vstack([
        _interpolate_static_motor_quantity(rpm_matrix[i], motor_spec, 'thrust_gf')
        for i in range(4)
    ])
    torque_reference = np.vstack([
        _interpolate_static_motor_quantity(rpm_matrix[i], motor_spec, 'torque_nm')
        for i in range(4)
    ])
    thrust_kgf = thrust_gf_reference * density_ratio / 1000.0
    prop_torque_nm = torque_reference * density_ratio

    # ArduPilot Quad-X: 0 front-right CCW, 1 rear-left CCW,
    # 2 front-left CW, 3 rear-right CW. Positive x is forward and positive y right.
    x_normalized = np.asarray([1.0, -1.0, 1.0, -1.0])
    y_normalized = np.asarray([1.0, -1.0, -1.0, 1.0])
    reaction_sign = np.asarray([-1.0, -1.0, 1.0, 1.0])
    total_thrust_kgf = np.nansum(thrust_kgf, axis=0)
    cg_forward_fraction = np.divide(
        np.nansum(thrust_kgf * x_normalized[:, None], axis=0), total_thrust_kgf,
        out=np.full(len(time_abs), np.nan), where=total_thrust_kgf > 0
    )
    cg_right_fraction = np.divide(
        np.nansum(thrust_kgf * y_normalized[:, None], axis=0), total_thrust_kgf,
        out=np.full(len(time_abs), np.nan), where=total_thrust_kgf > 0
    )
    tilt_vertical_factor = (
        np.cos(np.deg2rad(roll_deg)) * np.cos(np.deg2rad(pitch_deg))
    )
    estimated_mass_kg = total_thrust_kgf * tilt_vertical_factor
    yaw_reaction_nm = (
        np.nansum(prop_torque_nm * reaction_sign[:, None], axis=0)
        * tilt_vertical_factor
    )
    total_thrust_n = total_thrust_kgf * 9.80665
    pitch_moment_nm = (
        total_thrust_n * cg_forward_fraction * half_length_m
        if half_length_m is not None else np.full(len(time_abs), np.nan)
    )
    roll_moment_nm = (
        total_thrust_n * cg_right_fraction * half_width_m
        if half_width_m is not None else np.full(len(time_abs), np.nan)
    )

    valid_hover = np.zeros(len(time_abs), dtype=bool)
    window_rows = []
    for window_id, (start, end) in enumerate(segments, start=1):
        window_mask = hover_mask.copy()
        window_mask[:start] = False
        window_mask[end + 1:] = False
        valid_hover |= window_mask
        row = {
            'window': window_id,
            'start_s': float(display_time[start]),
            'end_s': float(display_time[end]),
            'duration_s': float(display_time[end] - display_time[start]),
            'estimated_mass_kg': float(np.nanmedian(estimated_mass_kg[window_mask])),
            'estimated_weight_lb': float(np.nanmedian(estimated_mass_kg[window_mask]) * 2.2046226218),
            'cg_forward_fraction': float(np.nanmedian(cg_forward_fraction[window_mask])),
            'cg_right_fraction': float(np.nanmedian(cg_right_fraction[window_mask])),
            'yaw_reaction_nm': float(np.nanmedian(yaw_reaction_nm[window_mask])),
            'median_roll_deg': float(np.nanmedian(roll_deg[window_mask])),
            'median_pitch_deg': float(np.nanmedian(pitch_deg[window_mask])),
            'median_altitude_m': float(np.nanmedian(relative_altitude[window_mask])),
            'median_vertical_speed_mps': float(np.nanmedian(down_velocity[window_mask])),
            'median_horizontal_speed_mps': float(np.nanmedian(horizontal_speed[window_mask])),
            'air_density_ratio': float(np.nanmedian(density_ratio[window_mask])),
        }
        for i in range(4):
            row[f'esc{i}_rpm'] = float(np.nanmedian(rpm_matrix[i, window_mask]))
            row[f'esc{i}_thrust_kgf'] = float(np.nanmedian(thrust_kgf[i, window_mask]))
        if half_length_m is not None:
            row['cg_forward_m'] = row['cg_forward_fraction'] * half_length_m
            row['pitch_moment_nm'] = float(np.nanmedian(pitch_moment_nm[window_mask]))
        if half_width_m is not None:
            row['cg_right_m'] = row['cg_right_fraction'] * half_width_m
            row['roll_moment_nm'] = float(np.nanmedian(roll_moment_nm[window_mask]))
        window_rows.append(row)

    return {
        'time': display_time,
        'hover_mask': valid_hover,
        'segments': segments,
        'windows': window_rows,
        'rpm': rpm_matrix,
        'thrust_kgf': thrust_kgf,
        'estimated_mass_kg': estimated_mass_kg,
        'cg_forward_fraction': cg_forward_fraction,
        'cg_right_fraction': cg_right_fraction,
        'yaw_reaction_nm': yaw_reaction_nm,
        'pitch_moment_nm': pitch_moment_nm,
        'roll_moment_nm': roll_moment_nm,
        'half_length_m': half_length_m,
        'half_width_m': half_width_m,
        'density_ratio': density_ratio,
        'motor_spec': motor_spec,
    }


def plot_hover_balance(
        filepath, esc_data, active_escs, title_prefix, save_path,
        motor_spec, hover_config=None):
    """Plot sustained-hover thrust balance, CG estimate, moments, and weight."""
    result = analyze_hover_balance(
        filepath, esc_data, active_escs, motor_spec, hover_config
    )
    setup_style()
    if result.get('error'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, result['error'], ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        fig.suptitle(f'{title_prefix} - Hover Balance / CG Estimate')
        fig.savefig(save_path, dpi=140)
        print(result['error'])
        print(f'Saved: {save_path}')
        plt.show()
        return

    windows = result['windows']
    summary_csv = os.path.splitext(save_path)[0] + '_windows.csv'
    pd.DataFrame(windows).to_csv(summary_csv, index=False)
    print(f'Saved hover-window table: {summary_csv}')

    time = result['time']
    hover_mask = result['hover_mask']
    masked = lambda values: np.where(hover_mask, values, np.nan)
    total_hover_duration = sum(row['duration_s'] for row in windows)
    median_mass = float(np.nanmedian(result['estimated_mass_kg'][hover_mask]))
    median_weight_lb = median_mass * 2.2046226218
    median_forward = float(np.nanmedian(result['cg_forward_fraction'][hover_mask]))
    median_right = float(np.nanmedian(result['cg_right_fraction'][hover_mask]))
    median_yaw = float(np.nanmedian(result['yaw_reaction_nm'][hover_mask]))
    esc_labels = ['ESC 0 (front right)', 'ESC 1 (back left)',
                  'ESC 2 (front left)', 'ESC 3 (back right)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        f'{title_prefix} - Sustained Hover Balance / CG Estimate\n'
        f'{total_hover_duration:.1f}s accepted | Weight {median_mass:.2f} kg '
        f'({median_weight_lb:.1f} lb) | CG {median_forward * 100:+.1f}% forward, '
        f'{median_right * 100:+.1f}% right',
        fontsize=14
    )

    ax = axes[0, 0]
    for i in range(4):
        ax.plot(time, masked(result['rpm'][i]), color=COLORS[i % 4],
                linewidth=1.4, label=esc_labels[i])
    ax.set_xlabel('Elapsed time in selected run (s)')
    ax.set_ylabel('Motor speed (RPM)')
    ax.set_title('RPM during accepted hover')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for i in range(4):
        ax.plot(time, masked(result['thrust_kgf'][i]), color=COLORS[i % 4],
                linewidth=1.4, label=esc_labels[i])
    ax.set_xlabel('Elapsed time in selected run (s)')
    ax.set_ylabel('Estimated static thrust (kgf)')
    ax.set_title('RPM-mapped thrust during hover')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    motor_x = np.asarray([1.0, -1.0, -1.0, 1.0])  # Plot horizontal: right-positive.
    motor_y = np.asarray([1.0, -1.0, 1.0, -1.0])  # Plot vertical: forward-positive.
    ax.scatter(motor_x, motor_y, s=90, color=[COLORS[i % 4] for i in range(4)])
    for i in range(4):
        ax.annotate(esc_labels[i], (motor_x[i], motor_y[i]), xytext=(5, 5),
                    textcoords='offset points')
    cg_x = result['cg_right_fraction'][hover_mask]
    cg_y = result['cg_forward_fraction'][hover_mask]
    ax.plot(cg_x, cg_y, color=COLORS[1], linewidth=1.0, alpha=0.55)
    ax.scatter([median_right], [median_forward], color=COLORS[3], marker='x',
               s=100, linewidths=2.5, zorder=4)
    ax.annotate(
        'Median CG', (median_right, median_forward), xytext=(8, -14),
        textcoords='offset points', color=COLORS[3]
    )
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Right-heavy offset / half motor width')
    ax.set_ylabel('Nose-heavy offset / half motor length')
    ax.set_title('Thrust centroid in Quad-X frame')
    ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    ax.plot(time, masked(result['cg_forward_fraction'] * 100), color=COLORS[0],
            linewidth=1.4, label='Pitch balance: forward %')
    ax.plot(time, masked(result['cg_right_fraction'] * 100), color=COLORS[2],
            linewidth=1.4, label='Roll balance: right %')
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Elapsed time in selected run (s)')
    ax.set_ylabel('Normalized thrust moment / CG offset (%)')
    yaw_axis = ax.twinx()
    yaw_axis.plot(time, masked(result['yaw_reaction_nm']), color=COLORS[3],
                  linewidth=1.2, alpha=0.75, label='Yaw reaction torque')
    yaw_axis.set_ylabel('Estimated yaw reaction torque (N m)')
    lines = ax.get_lines()[:2] + yaw_axis.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='best')
    ax.set_title(
        f'Median moments: pitch {median_forward * 100:+.1f}%, '
        f'roll {median_right * 100:+.1f}%, yaw {median_yaw:+.3f} N m'
    )
    ax.grid(True, alpha=0.25)

    geometry_note = 'Pitch/roll are normalized because motor half-spacing is not configured.'
    if result['half_length_m'] is not None and result['half_width_m'] is not None:
        pitch_nm = float(np.nanmedian(result['pitch_moment_nm'][hover_mask]))
        roll_nm = float(np.nanmedian(result['roll_moment_nm'][hover_mask]))
        geometry_note = f'Configured geometry gives pitch {pitch_nm:+.2f} N m and roll {roll_nm:+.2f} N m.'
    fig.text(
        0.5, 0.012,
        f"Static map: {motor_spec.get('static_test_reference', motor_spec['prop'])}. "
        f"Thrust corrected by median local pressure; assumed air temperature "
        f"{float((hover_config or {}).get('air_temperature_c', 25.0)):g}C. {geometry_note}",
        ha='center', fontsize=9
    )
    fig.tight_layout(rect=[0.01, 0.045, 0.99, 0.92])
    fig.savefig(save_path, dpi=140)
    print(f'Saved hover balance analysis: {save_path}')
    print(
        f'Hover estimate: {median_mass:.2f} kg ({median_weight_lb:.1f} lb), '
        f'CG {median_forward * 100:+.2f}% forward / {median_right * 100:+.2f}% right, '
        f'yaw reaction {median_yaw:+.3f} N m.'
    )
    plt.show()

