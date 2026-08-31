"""Derived propulsion metrics and run-level aggregation."""

from collections import defaultdict

import numpy as np

from . import runtime
from .config import get_current_scale
from .constants import DEFAULT_MIN_THROTTLE_THRESHOLD
from .motors import (
    DEFAULT_MOTOR_SPEC_KEY,
    calculate_motor_efficiency,
    calculate_propeller_constant,
    get_motor_spec,
)
from .telemetry import _interpolate_without_bridging_gaps

def lowpass_filter(data, times, rc_constant):
    """Apply exponential moving average (low-pass filter) to data.
    
    Args:
        data: List of values to filter
        times: List of timestamps (seconds)
        rc_constant: RC time constant in seconds (0 = no filtering)
    
    Returns:
        Filtered data as list
    """
    if rc_constant <= 0 or len(data) < 2:
        return list(data)
    
    filtered = [data[0]]
    for i in range(1, len(data)):
        dt = times[i] - times[i-1] if i < len(times) else 0.1
        if dt <= 0:
            dt = 0.1  # Default time step
        alpha = dt / (rc_constant + dt)  # EMA smoothing factor
        filtered.append(alpha * data[i] + (1 - alpha) * filtered[-1])
    
    return filtered

def build_max_throttle_series(esc_data, ref_times):
    """Build a max-throttle series aligned to the reference time base."""
    max_throttle = [0.0] * len(ref_times)
    for inst, data in esc_data.items():
        throttle = data.get('throttle', [])
        for j in range(min(len(throttle), len(ref_times))):
            if throttle[j] > max_throttle[j]:
                max_throttle[j] = throttle[j]
    return max_throttle


def build_active_esc_count_series(esc_data, ref_times, throttle_threshold):
    """Count active ESCs per time index based on per-ESC throttle."""
    counts = []
    for j in range(len(ref_times)):
        cnt = 0
        for _, data in esc_data.items():
            throttle = data.get('throttle', [])
            if j < len(throttle) and throttle[j] >= throttle_threshold:
                cnt += 1
        counts.append(cnt)
    return counts

def build_synchronized_total_series(esc_data, per_esc, selected_escs=None):
    """Synchronize ESCs by timestamp and aggregate only currently reporting ESCs.

    Voltage is the average of the selected ESCs that are reporting at each time.
    Current is their sum. `power` is average voltage times summed current, while
    `sum_esc_power` is the independent sum of each ESC's V*I for comparison.
    """
    selected = sorted(selected_escs if selected_escs is not None else esc_data.keys())
    selected = [i for i in selected if i in esc_data and len(esc_data[i].get('time', [])) >= 2]
    if not selected:
        return {
            'time': [], 'curr': [], 'power': [], 'sum_esc_power': [],
            'avg_volt': [], 'avg_temp': [], 'esc_count': []
        }

    all_times = [
        np.asarray(esc_data[i]['time'], dtype=float)
        for i in selected if esc_data[i].get('time')
    ]
    positive_steps = np.concatenate([
        np.diff(t)[np.diff(t) > 0] for t in all_times if len(t) >= 2
    ])
    grid_dt = float(np.median(positive_steps)) if len(positive_steps) else 0.25
    grid_dt = min(1.0, max(0.05, grid_dt))
    grid_start = min(float(t[0]) for t in all_times)
    grid_end = max(float(t[-1]) for t in all_times)
    grid = np.arange(grid_start, grid_end + grid_dt * 0.5, grid_dt)

    volt_rows, curr_rows, temp_rows = [], [], []
    for i in selected:
        data = esc_data[i]
        deriv = per_esc.get(i, {})
        volt_rows.append(_interpolate_without_bridging_gaps(
            data['time'], deriv.get('volt_filtered', data.get('volt', [])), grid
        ))
        curr_rows.append(_interpolate_without_bridging_gaps(
            data['time'], deriv.get('curr_filtered', data.get('curr', [])), grid
        ))
        temp_rows.append(_interpolate_without_bridging_gaps(
            data['time'], data.get('temp', []), grid
        ))

    volts = np.vstack(volt_rows)
    currents = np.vstack(curr_rows)
    temps = np.vstack(temp_rows)
    reporting = np.isfinite(volts) & np.isfinite(currents)
    esc_count = np.sum(reporting, axis=0)
    voltage_sum = np.nansum(np.where(reporting, volts, np.nan), axis=0)
    avg_volt = np.divide(
        voltage_sum, esc_count,
        out=np.full(len(grid), np.nan), where=esc_count > 0
    )
    temp_reporting = reporting & np.isfinite(temps)
    temp_count = np.sum(temp_reporting, axis=0)
    temp_sum = np.nansum(np.where(temp_reporting, temps, np.nan), axis=0)
    avg_temp = np.divide(
        temp_sum, temp_count,
        out=np.full(len(grid), np.nan), where=temp_count > 0
    )
    total_curr = np.nansum(np.where(reporting, currents, np.nan), axis=0)
    sum_esc_power = np.nansum(np.where(reporting, volts * currents, np.nan), axis=0)
    total_curr[esc_count == 0] = np.nan
    sum_esc_power[esc_count == 0] = np.nan
    avg_power = avg_volt * total_curr

    return {
        'time': grid.tolist(),
        'curr': total_curr.tolist(),
        'power': avg_power.tolist(),
        'sum_esc_power': sum_esc_power.tolist(),
        'avg_volt': avg_volt.tolist(),
        'avg_temp': avg_temp.tolist(),
        'esc_count': esc_count.astype(int).tolist(),
    }


def compute_derived_metrics(esc_data, rc_filter=0, prop_k=None, current_scale_rules=None):
    """Compute power, efficiency, and aggregate metrics.
    
    Args:
        esc_data: Dict of ESC data by instance
        rc_filter: Low-pass filter RC constant (0 = no filtering)
        prop_k: Propeller constant for efficiency calculation
        current_scale_rules: Optional list of current scale rules
    """
    if prop_k is None:
        prop_k = calculate_propeller_constant(get_motor_spec(DEFAULT_MOTOR_SPEC_KEY))
    current_scale_rules = current_scale_rules or []

    derived = {
        'per_esc': {},  # Power and efficiency per ESC
        'total': {
            'time': [], 'curr': [], 'power': [], 'sum_esc_power': [],
            'avg_volt': [], 'avg_temp': [], 'esc_count': []
        }
    }
    
    # First, find a common time base (use first ESC with data as reference)
    ref_instance = None
    if esc_data:
        # Prefer ESC 0 if it has data, otherwise take first one with data
        if 0 in esc_data and len(esc_data[0]['time']) > 0:
            ref_instance = 0
        else:
            for inst in sorted(esc_data.keys()):
                if len(esc_data[inst]['time']) > 0:
                    ref_instance = inst
                    break
        
    if ref_instance is None:
        return derived
    
    ref_times = esc_data[ref_instance]['time']

    max_throttle = build_max_throttle_series(esc_data, ref_times)
    use_scale_rules = bool(current_scale_rules) and max(max_throttle) > 0
    if current_scale_rules and not use_scale_rules:
        print("Warning: Current scale rules configured but throttle data is missing. Skipping scaling.")
    
    for inst in esc_data:
        data = esc_data[inst]
        n = len(data['time'])
        times = data['time']
        
        # Apply low-pass filter to voltage and current if enabled
        if rc_filter > 0:
            volt_filtered = lowpass_filter(data['volt'], times, rc_filter)
            curr_filtered = lowpass_filter(data['curr'], times, rc_filter)
        else:
            volt_filtered = data['volt']
            curr_filtered = data['curr']
        
        power = []
        efficiency = []
        efficiency_rpm_w = []
        curr_scaled = []
        
        for j in range(n):
            v = volt_filtered[j]
            i_raw = curr_filtered[j]
            rpm = data['rpm'][j]

            throttle_pwm = max_throttle[j] if j < len(max_throttle) else (max_throttle[-1] if max_throttle else 0)
            scale = get_current_scale(throttle_pwm, current_scale_rules) if use_scale_rules else 1.0
            i = i_raw * scale
            curr_scaled.append(i)
            
            p = v * i  # Input Power (Watts)
            power.append(p)
            
            # Motor Efficiency: Output Power / Input Power (%)
            # Uses propeller cubic relationship: P_out = k × RPM³
            # Filter out low current data (sensor inaccurate below threshold)
            if i >= runtime.options.min_current_threshold and p > 1.0:
                eff_pct = calculate_motor_efficiency(rpm, p, prop_k)
                eff_rpm_w = rpm / p if p > 0 else 0
            else:
                eff_pct = None  # Mark as invalid/unreliable
                eff_rpm_w = None
            
            efficiency.append(eff_pct)  # Default to %
            efficiency_rpm_w.append(eff_rpm_w)
            
        derived['per_esc'][inst] = {
            'power': power,
            'efficiency': efficiency,
            'efficiency_pct': efficiency,
            'efficiency_rpm_w': efficiency_rpm_w,
            'volt_filtered': volt_filtered,
            'curr_filtered': curr_scaled
        }
    
    derived['total'] = build_synchronized_total_series(esc_data, derived['per_esc'])
    
    return derived


def analyze_runs_from_cache(runs, all_esc_data, prop_k=None, current_scale_rules=None):
    """Compute summary stats for each run using cached data (fast).
    
    Args:
        runs: List of (start_us, end_us) tuples
        all_esc_data: Pre-loaded ESC data from cache
    """
    stats = []
    
    for idx, (start, end) in enumerate(runs):
        # Filter cached data for this run's time range
        run_data = defaultdict(lambda: {'time_us': [], 'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': [], 'throttle': []})
        
        first_time = None
        for inst, data in all_esc_data.items():
            for i, t_us in enumerate(data['time_us']):
                if start and t_us < start:
                    continue
                if end and t_us > end:
                    break
                
                if first_time is None:
                    first_time = t_us
                
                run_start = start if start else first_time
                t_sec = (t_us - run_start) / 1e6
                
                run_data[inst]['time_us'].append(t_us)
                run_data[inst]['time'].append(t_sec)
                run_data[inst]['rpm'].append(data['rpm'][i])
                run_data[inst]['volt'].append(data['volt'][i])
                run_data[inst]['curr'].append(data['curr'][i])
                run_data[inst]['temp'].append(data['temp'][i])
                run_data[inst]['throttle'].append(data['throttle'][i] if 'throttle' in data and i < len(data['throttle']) else 0)
        
        derived = compute_derived_metrics(run_data, prop_k=prop_k, current_scale_rules=current_scale_rules)
        
        duration = (end - start) / 1e6 if (end and start) else 0
        max_rpm = max_curr = max_temp = avg_eff = total_energy = total_charge_ah = 0
        
        for inst in run_data:
            d = run_data[inst]
            if d['rpm']:
                max_rpm = max(max_rpm, max(d['rpm']))
            curr_vals = derived['per_esc'].get(inst, {}).get('curr_filtered', d.get('curr', []))
            if curr_vals:
                max_curr = max(max_curr, max(curr_vals))
            if d['temp']:
                max_temp = max(max_temp, max(d['temp']))
        
        # Integrate each ESC on its own timestamp axis, then sum. ESC message
        # streams can have unequal lengths and small timestamp offsets; summing
        # samples by array index before integration can materially overstate Wh.
        for inst, d in run_data.items():
            times = d.get('time', [])
            powers = derived['per_esc'].get(inst, {}).get('power', [])
            currents = derived['per_esc'].get(inst, {}).get('curr_filtered', d.get('curr', []))
            n_power = min(len(times), len(powers))
            n_current = min(len(times), len(currents))
            if n_power >= 2:
                energy_ws = sum(
                    (powers[i] + powers[i + 1]) / 2 * (times[i + 1] - times[i])
                    for i in range(n_power - 1)
                )
                total_energy += energy_ws / 3600
            if n_current >= 2:
                charge_as = sum(
                    (currents[i] + currents[i + 1]) / 2 * (times[i + 1] - times[i])
                    for i in range(n_current - 1)
                )
                total_charge_ah += charge_as / 3600

        # Average efficiency across all ESCs (only valid readings)
        eff_values = []
        for inst in derived['per_esc']:
            eff_values.extend([e for e in derived['per_esc'][inst]['efficiency'] if e is not None and e > 0])
        if eff_values:
            avg_eff = sum(eff_values) / len(eff_values)
        
        stats.append({
            'id': idx + 1,
            'start': start,
            'end': end,
            'duration': duration,
            'max_rpm': max_rpm,
            'max_curr': max_curr,
            'max_temp': max_temp,
            'energy_wh': total_energy,
            'charge_ah': total_charge_ah,
            'avg_eff': avg_eff
        })
        
    return stats


def combine_runs_data(runs, all_esc_data, current_threshold=None, throttle_threshold=DEFAULT_MIN_THROTTLE_THRESHOLD):
    """Combine all runs into continuous data without gaps between runs.
    
    Args:
        runs: List of (start_us, end_us) tuples
        all_esc_data: Pre-loaded ESC data from cache
        current_threshold: Only include data where current >= threshold
        throttle_threshold: Only include data where throttle >= threshold (filters shutdown data)
    
    Returns:
        combined_esc_data: Dict with continuous time axis, run boundaries marked
        run_boundaries: List of cumulative times where each run ends
    """
    if current_threshold is None:
        current_threshold = runtime.options.min_current_threshold

    combined = defaultdict(lambda: {'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': [], 'throttle': []})
    run_boundaries = []
    cumulative_time = 0.0
    
    for run_idx, (start, end) in enumerate(runs):
        if start == 0 and end == 0:
            continue  # Skip "no runs" marker
        
        run_duration = 0.0
        run_data_added = False
        
        for inst, data in all_esc_data.items():
            throttle_data = data.get('throttle', [0] * len(data['time_us']))
            
            for i, t_us in enumerate(data['time_us']):
                if t_us < start:
                    continue
                if t_us > end:
                    break
                
                curr = data['curr'][i]
                throttle = throttle_data[i] if i < len(throttle_data) else 0
                
                # Filter by both current AND throttle thresholds
                # This eliminates both low-current and shutdown/ramp-down data
                if curr < current_threshold:
                    continue
                if throttle_threshold > 0 and throttle > 0 and throttle < throttle_threshold:
                    continue
                
                # Calculate time relative to run start, then add cumulative offset
                t_relative = (t_us - start) / 1e6
                t_continuous = cumulative_time + t_relative
                
                combined[inst]['time'].append(t_continuous)
                combined[inst]['rpm'].append(data['rpm'][i])
                combined[inst]['volt'].append(data['volt'][i])
                combined[inst]['curr'].append(data['curr'][i])
                combined[inst]['temp'].append(data['temp'][i])
                combined[inst]['throttle'].append(throttle)
                
                run_duration = max(run_duration, t_relative)
                run_data_added = True
        
        if run_data_added:
            cumulative_time += run_duration + 0.5  # Small gap between runs for visual separation
            run_boundaries.append(cumulative_time - 0.25)  # Mark boundary in middle of gap
    
    return combined, run_boundaries
