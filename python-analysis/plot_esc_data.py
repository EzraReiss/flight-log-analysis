#!/usr/bin/env python3
"""
Interactive ESC Analysis Tool for ArduPilot .bin logs
Features: Run detection, Power/Efficiency metrics, Voltage Sag analysis, ESC filtering
Caching: Parses bin file once and caches to Parquet for fast subsequent loads
"""

import sys
import os
import csv
import json
import hashlib
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is not installed. Run: pip install pandas")
    sys.exit(1)

try:
    from pymavlink import mavutil
except ImportError:
    print("Error: pymavlink is not installed. Run: pip install pymavlink")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Minimum current threshold for reliable sensor readings
# Data below this value is excluded from efficiency/voltage sag calculations
MIN_CURRENT_THRESHOLD = 10.0  # Amps (raised from 10A to exclude shutdown periods)

# Minimum throttle threshold for active motor data
# Data below this PWM value indicates motor is ramping down/stopped
MIN_THROTTLE_THRESHOLD = 1400  # PWM value (typically 1000-2000 range)

# Low-pass filter RC time constant (seconds)
# Higher = more smoothing, lower = less smoothing
# Range: 0.25 to 5.0 seconds (0 = disabled)
DEFAULT_FILTER_RC = 2.0

# Cache directory (stored next to the bin file)
CACHE_VERSION = "v4"  # Increment when cache format changes (v4 fixes throttle lookup to use max)


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

# =============================================================================
# Motor Specification Data (for benchmark comparison)
# =============================================================================

# MAD V62 PRO IPE 210KV with CF FLUXER 22.1x7.4 VTOL prop, AMPX 80A ESC, 12S
# Source: Manufacturer datasheet
MOTOR_SPEC = {
    'name': 'MAD V62 PRO IPE 210KV (12S)',
    'prop': 'CF FLUXER 22.1x7.4 VTOL',
    'data': {
        # Throttle %: [Voltage, Current, Input Power, Output Power, RPM, Efficiency %]
        30:  [47.76, 3.62,  172.5,  128.0, 2662, 74.15],
        35:  [47.76, 4.96,  236.2,  180.6, 3007, 76.4],
        40:  [47.76, 6.88,  328.0,  256.8, 3375, 78.23],
        45:  [47.72, 9.66,  460.5,  367.2, 3798, 79.69],
        50:  [47.69, 12.97, 618.2,  498.6, 4192, 80.6],
        55:  [47.68, 16.13, 768.4,  622.9, 4537, 81.02],
        60:  [47.61, 20.52, 976.5,  763.2, 4834, 78.11],
        65:  [47.63, 24.21, 1152.9, 910.9, 5118, 78.97],
        70:  [47.52, 27.93, 1326.7, 1069.3, 5400, 80.56],
        75:  [47.55, 33.46, 1590.3, 1240.2, 5647, 77.96],
        80:  [47.47, 38.38, 1821.6, 1419.5, 5913, 77.9],
        85:  [47.37, 45.41, 2150.5, 1635.4, 6181, 76.01],
        90:  [47.34, 50.03, 2368.1, 1830.0, 6390, 77.24],
        95:  [47.26, 58.14, 2747.5, 2016.1, 6611, 73.37],
        100: [47.17, 59.3,  2846.4, 2058.2, 6793, 72.31],
    }
}

# Calculate propeller constant k from datasheet: P_out = k × RPM³
# For propellers: Output Power is proportional to RPM cubed
def calculate_propeller_constant():
    """Derive propeller constant k from datasheet where P_out = k × RPM³."""
    k_values = []
    for throttle, data in MOTOR_SPEC['data'].items():
        output_power = data[3]  # Output Power (W)
        rpm = data[4]           # RPM
        if rpm > 0:
            k = output_power / (rpm ** 3)
            k_values.append(k)
    return sum(k_values) / len(k_values) if k_values else 0

# Propeller constant for CF FLUXER 22.1x7.4 VTOL prop
# P_output (W) = PROP_K * RPM^3
PROP_K = calculate_propeller_constant()

def estimate_output_power(rpm):
    """Estimate propeller output power from RPM using cubic relationship.
    
    For propellers: P_out = k × RPM³
    Returns estimated output power in Watts.
    """
    if rpm <= 0:
        return 0
    return PROP_K * (rpm ** 3)

def calculate_motor_efficiency(rpm, input_power):
    """Calculate motor efficiency as Output Power / Input Power.
    
    Uses the propeller cubic relationship to estimate output power from RPM.
    Returns efficiency as percentage, or None if invalid.
    
    Note: Values >100% indicate the propeller constant may need calibration
    for the specific test conditions (voltage, temperature, etc.)
    """
    if input_power <= 0 or rpm <= 0:
        return None
    output_power = estimate_output_power(rpm)
    efficiency = (output_power / input_power) * 100
    return efficiency

# =============================================================================
# Caching Functions
# =============================================================================

def get_output_dir(filepath):
    """Get/create an organized output folder for this BIN file."""
    base_dir = os.path.dirname(filepath)
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    output_dir = os.path.join(base_dir, f"{base_name}_analysis")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_cache_path(filepath):
    """Get the cache file path for a given bin file."""
    output_dir = get_output_dir(filepath)
    return os.path.join(output_dir, "esc_data_cache.csv")

def get_cache_meta_path(filepath):
    """Get the cache metadata file path."""
    output_dir = get_output_dir(filepath)
    return os.path.join(output_dir, "cache_meta.json")

def is_cache_valid(filepath):
    """Check if cached data exists and is still valid."""
    cache_path = get_cache_path(filepath)
    meta_path = get_cache_meta_path(filepath)
    
    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check version
        if meta.get('version') != CACHE_VERSION:
            print("Cache version mismatch, reparsing...")
            return False
        
        # Check file modification time
        bin_mtime = os.path.getmtime(filepath)
        if meta.get('bin_mtime') != bin_mtime:
            print("Source file modified, reparsing...")
            return False
        
        # Check file size
        bin_size = os.path.getsize(filepath)
        if meta.get('bin_size') != bin_size:
            print("Source file size changed, reparsing...")
            return False
            
        return True
    except Exception as e:
        print(f"Cache validation error: {e}")
        return False

def load_from_cache(filepath):
    """Load ESC data and runs from cache (CSV format)."""
    cache_path = get_cache_path(filepath)
    meta_path = get_cache_meta_path(filepath)
    
    print("Loading from cache...")
    
    try:
        # Load ESC data from CSV
        df = pd.read_csv(cache_path)
        
        # Load metadata (runs)
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        runs = [tuple(r) for r in meta.get('runs', [])]
        
        # Convert DataFrame back to esc_data dict format
        esc_data = defaultdict(lambda: {'time_us': [], 'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': [], 'throttle': []})
        
        for inst in df['instance'].unique():
            inst_df = df[df['instance'] == inst].sort_values('time')
            esc_data[inst] = {
                'time_us': inst_df['time_us'].tolist(),
                'time': inst_df['time'].tolist(),
                'rpm': inst_df['rpm'].tolist(),
                'volt': inst_df['volt'].tolist(),
                'curr': inst_df['curr'].tolist(),
                'temp': inst_df['temp'].tolist(),
                'throttle': inst_df['throttle'].tolist() if 'throttle' in inst_df.columns else [0] * len(inst_df)
            }
        
        print(f"Loaded {len(df)} cached data points, {len(runs)} runs")
        return esc_data, runs
        
    except Exception as e:
        print(f"Cache load error: {e}")
        return None, None

def save_to_cache(filepath, esc_data, runs):
    """Save ESC data and runs to cache (CSV format for easy viewing)."""
    cache_path = get_cache_path(filepath)
    meta_path = get_cache_meta_path(filepath)
    
    print("Saving to cache...")
    
    try:
        # Convert esc_data to DataFrame
        rows = []
        for inst, data in esc_data.items():
            for i in range(len(data['time'])):
                rows.append({
                    'instance': inst,
                    'time_us': data['time_us'][i],
                    'time': data['time'][i],
                    'rpm': data['rpm'][i],
                    'volt': data['volt'][i],
                    'curr': data['curr'][i],
                    'temp': data['temp'][i],
                    'throttle': data['throttle'][i] if 'throttle' in data and i < len(data['throttle']) else 0
                })
        
        df = pd.DataFrame(rows)
        # Save as CSV for human readability
        df.to_csv(cache_path, index=False)
        
        # Save metadata
        meta = {
            'version': CACHE_VERSION,
            'bin_mtime': os.path.getmtime(filepath),
            'bin_size': os.path.getsize(filepath),
            'runs': runs,
            'esc_count': len(esc_data)
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Cached {len(df)} data points to {os.path.basename(cache_path)}")
        
    except Exception as e:
        print(f"Cache save error: {e}")

# =============================================================================
# Data Loading Functions
# =============================================================================

def detect_runs(filepath, throttle_threshold=1200, cooldown_sec=10.0):
    """Scans log for RCOU messages to identify active runs.
    
    Uses max throttle rather than average to handle 2-ESC and 4-ESC configurations.
    """
    try:
        mlog = mavutil.mavlink_connection(filepath)
    except Exception as e:
        print(f"Error opening file: {e}")
        return []

    runs = []
    in_run = False
    current_run_start = 0
    potential_end_time = 0
    last_msg_time = 0
    
    while True:
        try:
            msg = mlog.recv_match(type=['RCOU'])
            if msg is None:
                break
            
            last_msg_time = msg.TimeUS
            # Get throttle values from channels 1-8 (covers most ESC configurations)
            # Use max throttle to detect activity (works for 2, 4, 6, 8 ESC setups)
            throttles = [getattr(msg, f'C{i}', 0) for i in range(1, 9)]
            # Filter out invalid/unused channels (typically 0 or very low values)
            active_throttles = [t for t in throttles if t > 900]
            max_throttle = max(active_throttles) if active_throttles else 0
            
            if not in_run:
                if max_throttle > throttle_threshold:
                    in_run = True
                    current_run_start = msg.TimeUS
                    potential_end_time = 0
            else:
                if max_throttle < throttle_threshold:
                    if potential_end_time == 0:
                        potential_end_time = msg.TimeUS
                    elif (msg.TimeUS - potential_end_time) > (cooldown_sec * 1e6):
                        runs.append((current_run_start, msg.TimeUS))
                        in_run = False
                        potential_end_time = 0
                else:
                    potential_end_time = 0
        except:
            continue
            
    if in_run:
        runs.append((current_run_start, last_msg_time))
         
    return runs


def load_esc_data(filepath, start_us=0, end_us=0):
    """Load ESC data within optional time range. Returns dict by instance.
    
    Also captures throttle (PWM) from RCOU messages and assigns to each ESC.
    """
    esc_data = defaultdict(lambda: {'time_us': [], 'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': [], 'throttle': []})
    
    try:
        mlog = mavutil.mavlink_connection(filepath)
    except:
        return esc_data
    
    # First pass: build throttle lookup from RCOU messages
    # Maps timestamp -> {esc_instance: throttle_pwm}
    throttle_lookup = {}
    
    while True:
        try:
            msg = mlog.recv_match(type=['RCOU'])
            if msg is None:
                break
            
            if start_us and msg.TimeUS < start_us:
                continue
            if end_us and msg.TimeUS > end_us:
                break
            
            # Map C1-C8 to ESC instances 0-7
            throttle_lookup[msg.TimeUS] = {
                0: getattr(msg, 'C1', 0),
                1: getattr(msg, 'C2', 0),
                2: getattr(msg, 'C3', 0),
                3: getattr(msg, 'C4', 0),
                4: getattr(msg, 'C5', 0),
                5: getattr(msg, 'C6', 0),
                6: getattr(msg, 'C7', 0),
                7: getattr(msg, 'C8', 0),
            }
            # Also store max throttle for simpler filtering (ESC instance may not match channel)
            channels = [throttle_lookup[msg.TimeUS][i] for i in range(8)]
            throttle_lookup[msg.TimeUS]['max'] = max(c for c in channels if c > 0)
        except:
            continue
    
    # Build sorted list of throttle timestamps for fast lookup
    throttle_times = sorted(throttle_lookup.keys())
    
    def get_throttle(time_us, esc_instance):
        """Find closest MAX throttle value for given timestamp.
        
        Uses max throttle across all channels since ESC instance may not map
        directly to RCOU channel numbers (varies by frame configuration).
        """
        if not throttle_times:
            return 0
        # Binary search for closest timestamp
        import bisect
        idx = bisect.bisect_left(throttle_times, time_us)
        if idx == 0:
            t = throttle_times[0]
        elif idx == len(throttle_times):
            t = throttle_times[-1]
        else:
            # Pick closer of the two neighbors
            t_before = throttle_times[idx - 1]
            t_after = throttle_times[idx]
            t = t_before if (time_us - t_before) < (t_after - time_us) else t_after
        
        # Return max throttle across all channels
        return throttle_lookup.get(t, {}).get('max', 0)
    
    # Second pass: read ESC messages with throttle lookup
    mlog = mavutil.mavlink_connection(filepath)
    first_time = None
        
    while True:
        try:
            msg = mlog.recv_match(type=['ESC'])
            if msg is None:
                break
            
            if start_us and msg.TimeUS < start_us:
                continue
            if end_us and msg.TimeUS > end_us:
                break
            
            if first_time is None:
                first_time = msg.TimeUS
                
            t_sec = (msg.TimeUS - first_time) / 1e6
            i = msg.Instance
            
            esc_data[i]['time_us'].append(msg.TimeUS)
            esc_data[i]['time'].append(t_sec)
            esc_data[i]['rpm'].append(msg.RPM)
            esc_data[i]['volt'].append(msg.Volt)
            esc_data[i]['curr'].append(msg.Curr)
            esc_data[i]['temp'].append(msg.Temp)
            esc_data[i]['throttle'].append(get_throttle(msg.TimeUS, i))
        except:
            continue
            
    return esc_data


def compute_derived_metrics(esc_data, rc_filter=0):
    """Compute power, efficiency, and aggregate metrics.
    
    Args:
        esc_data: Dict of ESC data by instance
        rc_filter: Low-pass filter RC constant (0 = no filtering)
    """
    derived = {
        'per_esc': {},  # Power and efficiency per ESC
        'total': {'time': [], 'curr': [], 'power': []}
    }
    
    # First, find a common time base (use ESC 0's timestamps as reference)
    if not esc_data or 0 not in esc_data:
        ref_instance = list(esc_data.keys())[0] if esc_data else None
    else:
        ref_instance = 0
        
    if ref_instance is None:
        return derived
    
    ref_times = esc_data[ref_instance]['time']
    derived['total']['time'] = ref_times
    
    # Initialize totals
    total_curr = [0.0] * len(ref_times)
    total_power = [0.0] * len(ref_times)
    
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
        
        for j in range(n):
            v = volt_filtered[j]
            i = curr_filtered[j]
            rpm = data['rpm'][j]
            
            p = v * i  # Input Power (Watts)
            power.append(p)
            
            # Motor Efficiency: Output Power / Input Power (%)
            # Uses propeller cubic relationship: P_out = k × RPM³
            # Filter out low current data (sensor inaccurate below threshold)
            if i >= MIN_CURRENT_THRESHOLD and p > 1.0:
                eff = calculate_motor_efficiency(rpm, p)
            else:
                eff = None  # Mark as invalid/unreliable
            efficiency.append(eff)
            
            # Accumulate totals (only if same length)
            if j < len(total_curr):
                total_curr[j] += i
                total_power[j] += p
        
        derived['per_esc'][inst] = {
            'power': power,
            'efficiency': efficiency,
            'volt_filtered': volt_filtered,
            'curr_filtered': curr_filtered
        }
    
    derived['total']['curr'] = total_curr
    derived['total']['power'] = total_power
    
    return derived


def analyze_runs_from_cache(runs, all_esc_data):
    """Compute summary stats for each run using cached data (fast).
    
    Args:
        runs: List of (start_us, end_us) tuples
        all_esc_data: Pre-loaded ESC data from cache
    """
    stats = []
    
    for idx, (start, end) in enumerate(runs):
        # Filter cached data for this run's time range
        run_data = defaultdict(lambda: {'time_us': [], 'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': []})
        
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
        
        derived = compute_derived_metrics(run_data)
        
        duration = (end - start) / 1e6 if (end and start) else 0
        max_rpm = max_curr = max_temp = avg_eff = total_energy = 0
        
        for inst in run_data:
            d = run_data[inst]
            if d['rpm']:
                max_rpm = max(max_rpm, max(d['rpm']))
            if d['curr']:
                max_curr = max(max_curr, max(d['curr']))
            if d['temp']:
                max_temp = max(max_temp, max(d['temp']))
        
        # Total energy (Wh) = integral of power over time
        if derived['total']['time'] and derived['total']['power']:
            times = derived['total']['time']
            powers = derived['total']['power']
            energy_ws = sum((powers[i] + powers[i+1]) / 2 * (times[i+1] - times[i]) 
                           for i in range(len(times)-1) if i+1 < len(powers))
            total_energy = energy_ws / 3600  # Convert Ws to Wh
            
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
            'avg_eff': avg_eff
        })
        
    return stats


def combine_runs_data(runs, all_esc_data, current_threshold=MIN_CURRENT_THRESHOLD, throttle_threshold=MIN_THROTTLE_THRESHOLD):
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
    combined = defaultdict(lambda: {'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': []})
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
                
                run_duration = max(run_duration, t_relative)
                run_data_added = True
        
        if run_data_added:
            cumulative_time += run_duration + 0.5  # Small gap between runs for visual separation
            run_boundaries.append(cumulative_time - 0.25)  # Mark boundary in middle of gap
    
    return combined, run_boundaries


# =============================================================================
# Plotting Functions
# =============================================================================

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def setup_style():
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            pass


def get_active_time_range(esc_data, threshold=MIN_CURRENT_THRESHOLD):
    """Find time range where current exceeds threshold (crops startup/shutdown).
    
    Returns (start_time, end_time) in seconds, or (None, None) if no valid data.
    This helps avoid Y-axis scaling issues from 0-value startup/shutdown periods.
    """
    all_times_above = []
    
    for inst, data in esc_data.items():
        for t, c in zip(data['time'], data['curr']):
            if c >= threshold:
                all_times_above.append(t)
    
    if not all_times_above:
        return None, None
    
    return min(all_times_above), max(all_times_above)


def plot_all_runs_combined(combined_data, run_boundaries, active_escs, title_prefix, save_path):
    """Plot all runs combined in a continuous view without gaps.
    
    Args:
        combined_data: ESC data stitched together from all runs (from combine_runs_data)
        run_boundaries: List of times where run boundaries occur
        active_escs: List of ESC indices to show
        title_prefix: Title for the plot
        save_path: Path to save the plot
    """
    setup_style()
    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'{title_prefix} - All Runs Combined (Current >= {MIN_CURRENT_THRESHOLD}A)', fontsize=14)
    
    metrics = [('rpm', 'RPM'), ('volt', 'Voltage (V)'), ('curr', 'Current (A)'), ('temp', 'Temp (°C)')]
    
    for ax_idx, (key, ylabel) in enumerate(metrics):
        ax = axs[ax_idx]
        
        for i in active_escs:
            if i in combined_data and combined_data[i]['time']:
                values = combined_data[i][key]
                times = combined_data[i]['time']
                if values:
                    label = f'ESC {i} ({min(values):.1f}/{np.median(values):.1f}/{max(values):.1f})'
                    ax.plot(times, values, label=label, color=COLORS[i % 4], linewidth=1, alpha=0.8)
        
        # Add vertical lines at run boundaries
        for boundary in run_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    axs[3].set_xlabel('Combined Time (s)')
    
    # Add run labels at bottom
    if run_boundaries:
        for idx, boundary in enumerate(run_boundaries):
            start = 0 if idx == 0 else run_boundaries[idx-1]
            mid = (start + boundary) / 2
            axs[3].text(mid, axs[3].get_ylim()[0], f'R{idx+1}', ha='center', va='top', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_esc_basics(esc_data, active_escs, title_prefix, save_path, derived=None):
    """Plot RPM, Voltage, Current, Temperature for selected ESCs with per-ESC stats in legend.
    
    If derived dict is provided and contains filtered data, uses that for volt/curr plots.
    """
    setup_style()
    fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f'{title_prefix} - ESC Overview', fontsize=14)
    
    # Get active time range to crop out low-power startup/shutdown
    t_start, t_end = get_active_time_range(esc_data)
    
    metrics = [('rpm', 'RPM'), ('volt', 'Voltage (V)'), ('curr', 'Current (A)'), ('temp', 'Temp (°C)')]
    
    for ax_idx, (key, ylabel) in enumerate(metrics):
        ax = axs[ax_idx]
        
        for i in active_escs:
            if i in esc_data and esc_data[i]['time']:
                # Use filtered data for volt and curr if available
                if key == 'volt' and derived and i in derived.get('per_esc', {}) and 'volt_filtered' in derived['per_esc'][i]:
                    values = derived['per_esc'][i]['volt_filtered']
                elif key == 'curr' and derived and i in derived.get('per_esc', {}) and 'curr_filtered' in derived['per_esc'][i]:
                    values = derived['per_esc'][i]['curr_filtered']
                else:
                    values = esc_data[i][key]
                
                # Include stats in legend label (min/med/max)
                label = f'ESC {i} ({min(values):.1f}/{np.median(values):.1f}/{max(values):.1f})'
                ax.plot(esc_data[i]['time'], values, 
                       label=label, color=COLORS[i % 4], linewidth=1.2, alpha=0.85)
        
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # Crop X-axis to active range if available
        if t_start is not None and t_end is not None:
            ax.set_xlim(t_start, t_end)
    
    axs[3].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_power(esc_data, derived, active_escs, title_prefix, save_path):
    """Plot Total Current and Power (per ESC + Total) with stats."""
    setup_style()
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{title_prefix} - Power Analysis', fontsize=14)
    
    ref_time = derived['total']['time'] if derived['total']['time'] else []
    
    # Get active time range to crop out low-power startup/shutdown
    t_start, t_end = get_active_time_range(esc_data)
    
    # Total Current
    ax = axs[0]
    total_curr = derived['total']['curr']
    ax.plot(ref_time, total_curr, color='black', linewidth=2, label='Total')
    ax.set_ylabel('Total Current (A)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if total_curr:
        stats = f"Min: {min(total_curr):.1f}  Med: {np.median(total_curr):.1f}  Max: {max(total_curr):.1f}"
        ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Per-ESC Power
    ax = axs[1]
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            power = derived['per_esc'][i]['power']
            label = f'ESC {i} ({min(power):.0f}/{np.median(power):.0f}/{max(power):.0f})'
            ax.plot(esc_data[i]['time'], power,
                   label=label, color=COLORS[i % 4], linewidth=1.2, alpha=0.85)
    ax.set_ylabel('Power (W)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Total Power
    ax = axs[2]
    total_power = derived['total']['power']
    ax.plot(ref_time, total_power, color='purple', linewidth=2, label='Total Power')
    ax.set_ylabel('Total Power (W)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if total_power:
        stats = f"Min: {min(total_power):.1f}  Med: {np.median(total_power):.1f}  Max: {max(total_power):.1f}"
        ax.text(0.02, 0.95, stats, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Crop X-axis to active range
    if t_start is not None and t_end is not None:
        for ax in axs:
            ax.set_xlim(t_start, t_end)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_efficiency(esc_data, derived, active_escs, title_prefix, save_path):
    """Plot Efficiency (RPM/Watt) for selected ESCs. Only shows valid readings (current >= threshold)."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'{title_prefix} - Efficiency (RPM/Watt) [Current >= {MIN_CURRENT_THRESHOLD}A]', fontsize=14)
    
    # Get active time range for cropping
    t_start, t_end = get_active_time_range(esc_data)
    
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            times = esc_data[i]['time']
            effs = derived['per_esc'][i]['efficiency']
            # Filter out None values (unreliable low-current readings)
            valid_t = [t for t, e in zip(times, effs) if e is not None]
            valid_e = [e for e in effs if e is not None]
            if valid_t:
                label = f'ESC {i} ({min(valid_e):.1f}/{np.median(valid_e):.1f}/{max(valid_e):.1f})'
                ax.plot(valid_t, valid_e,
                       label=label, color=COLORS[i % 4], linewidth=1.2, alpha=0.85)
    
    ax.set_ylabel('Efficiency (RPM/W)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Crop X-axis to active range
    if t_start is not None and t_end is not None:
        ax.set_xlim(t_start, t_end)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_voltage_sag(esc_data, derived, active_escs, title_prefix, save_path, esc_count=4):
    """Scatter plot of Voltage vs TOTAL Current, colored by avg Temperature.
    
    Uses total current (sum of all ESCs) since battery voltage sag is a function
    of total system current draw, not individual ESC current.
    
    Args:
        esc_count: Number of ESCs in the system (for threshold calculation)
    """
    min_total_current = MIN_CURRENT_THRESHOLD * esc_count
    
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'{title_prefix} - Voltage Sag Analysis [Total Current >= {min_total_current}A]', fontsize=14)
    
    # Use total current from derived metrics and average voltage
    total_curr = derived['total']['curr']
    ref_times = derived['total']['time']
    
    if not total_curr or not ref_times:
        ax.text(0.5, 0.5, 'No data available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()
        return
    
    # Calculate average voltage and temperature across active ESCs at each time point
    n_points = len(ref_times)
    avg_volt = []
    avg_temp = []
    valid_curr = []
    
    for j in range(n_points):
        # Get voltage and temp from all active ESCs at this time index
        volts = []
        temps = []
        for i in active_escs:
            if i in esc_data and j < len(esc_data[i]['volt']):
                volts.append(esc_data[i]['volt'][j])
                temps.append(esc_data[i]['temp'][j])
        
        if volts and j < len(total_curr):
            curr = total_curr[j]
            # Filter by total current threshold (MIN_CURRENT_THRESHOLD * ESC count)
            if curr >= min_total_current:
                avg_volt.append(sum(volts) / len(volts))
                avg_temp.append(sum(temps) / len(temps))
                valid_curr.append(curr)
    
    if valid_curr and avg_volt:
        scatter = ax.scatter(valid_curr, avg_volt, c=avg_temp, cmap='coolwarm', 
                            alpha=0.5, s=10, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Avg Temperature (°C)')
        
        # Split data by median temperature and fit two trend lines
        if len(valid_curr) > 4:
            median_temp = np.median(avg_temp)
            
            # Cold data (below median temp)
            cold_mask = [t <= median_temp for t in avg_temp]
            cold_curr = [c for c, m in zip(valid_curr, cold_mask) if m]
            cold_volt = [v for v, m in zip(avg_volt, cold_mask) if m]
            
            # Hot data (above median temp)
            hot_mask = [t > median_temp for t in avg_temp]
            hot_curr = [c for c, m in zip(valid_curr, hot_mask) if m]
            hot_volt = [v for v, m in zip(avg_volt, hot_mask) if m]
            
            x_line = np.linspace(min(valid_curr), max(valid_curr), 100)
            
            # Cold trend line (blue)
            if len(cold_curr) > 2:
                z_cold = np.polyfit(cold_curr, cold_volt, 1)
                p_cold = np.poly1d(z_cold)
                ax.plot(x_line, p_cold(x_line), 'b-', linewidth=2, 
                       label=f'Cold (<{median_temp:.0f}°C): {z_cold[0]*1000:.2f}mV/A')
            
            # Hot trend line (red)
            if len(hot_curr) > 2:
                z_hot = np.polyfit(hot_curr, hot_volt, 1)
                p_hot = np.poly1d(z_hot)
                ax.plot(x_line, p_hot(x_line), 'r-', linewidth=2, 
                       label=f'Hot (>{median_temp:.0f}°C): {z_hot[0]*1000:.2f}mV/A')
            
            ax.legend(loc='upper right')
    else:
        ax.text(0.5, 0.5, f'No data above {min_total_current}A total threshold',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Total Current (A)')
    ax.set_ylabel('Battery Voltage (V)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_system_analysis(esc_data, derived, active_escs, title_prefix, save_path, esc_count):
    """Comprehensive system analysis showing voltage, current, power, and efficiency relationships.
    
    Creates 4 subplots:
    1. Voltage vs Efficiency (RPM/W) - colored by current
    2. Power vs Efficiency (RPM/W) - shows efficiency drop at high power
    3. Current vs Efficiency - per ESC scatter
    4. Voltage vs Total Power - shows power delivery at different voltage levels
    """
    min_total_current = MIN_CURRENT_THRESHOLD * esc_count
    
    setup_style()
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{title_prefix} - System Analysis [Current >= {min_total_current:.0f}A total]', fontsize=14)
    
    # Collect valid data points (above current threshold)
    all_volts = []
    all_curr = []
    all_power = []
    all_eff = []
    all_temp = []
    per_esc_data_local = {i: {'volt': [], 'curr': [], 'power': [], 'eff': []} for i in active_escs}
    
    for i in active_escs:
        if i not in esc_data or i not in derived['per_esc']:
            continue
        
        for j in range(len(esc_data[i]['curr'])):
            curr = esc_data[i]['curr'][j]
            if curr < MIN_CURRENT_THRESHOLD:
                continue
            
            volt = esc_data[i]['volt'][j]
            temp = esc_data[i]['temp'][j]
            power = derived['per_esc'][i]['power'][j] if j < len(derived['per_esc'][i]['power']) else 0
            eff = derived['per_esc'][i]['efficiency'][j] if j < len(derived['per_esc'][i]['efficiency']) else None
            
            if eff is not None and power > 0:
                all_volts.append(volt)
                all_curr.append(curr)
                all_power.append(power)
                all_eff.append(eff)
                all_temp.append(temp)
                
                per_esc_data_local[i]['volt'].append(volt)
                per_esc_data_local[i]['curr'].append(curr)
                per_esc_data_local[i]['power'].append(power)
                per_esc_data_local[i]['eff'].append(eff)
    
    if not all_volts:
        for ax in axs.flat:
            ax.text(0.5, 0.5, 'No valid data above threshold',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()
        return
    
    # --- Panel 1: Voltage vs Efficiency, colored by Current ---
    ax = axs[0, 0]
    scatter = ax.scatter(all_volts, all_eff, c=all_curr, cmap='plasma', alpha=0.5, s=15, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Current (A)')
    
    # Add trend line
    if len(all_volts) > 10:
        z = np.polyfit(all_volts, all_eff, 2)
        p = np.poly1d(z)
        v_range = np.linspace(min(all_volts), max(all_volts), 100)
        ax.plot(v_range, p(v_range), 'w-', linewidth=2, label='Trend')
        ax.plot(v_range, p(v_range), 'k--', linewidth=1)
    
    ax.set_xlabel('Battery Voltage (V)')
    ax.set_ylabel('Efficiency (RPM/W)')
    ax.set_title('Efficiency vs Voltage')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 2: Power vs Efficiency ---
    ax = axs[0, 1]
    scatter = ax.scatter(all_power, all_eff, c=all_volts, cmap='coolwarm', alpha=0.5, s=15, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Voltage (V)')
    
    ax.set_xlabel('Input Power (W)')
    ax.set_ylabel('Efficiency (RPM/W)')
    ax.set_title('Efficiency vs Power (colored by Voltage)')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 3: Current vs Efficiency per ESC ---
    ax = axs[1, 0]
    for i in active_escs:
        if per_esc_data_local[i]['curr']:
            ax.scatter(per_esc_data_local[i]['curr'], per_esc_data_local[i]['eff'], 
                      color=COLORS[i % 4], alpha=0.4, s=10, label=f'ESC {i}')
    
    ax.set_xlabel('ESC Current (A)')
    ax.set_ylabel('Efficiency (RPM/W)')
    ax.set_title('Efficiency vs Current (per ESC)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # --- Panel 4: Voltage vs Total Power ---
    ax = axs[1, 1]
    # Calculate total power for each voltage point
    total_power = derived['total']['power']
    ref_times = derived['total']['time']
    
    volt_power_pairs = []
    temp_for_pairs = []
    for j in range(min(len(ref_times), len(total_power))):
        # Get average voltage at this time
        volts = []
        temps = []
        for i in active_escs:
            if i in esc_data and j < len(esc_data[i]['volt']):
                volts.append(esc_data[i]['volt'][j])
                if j < len(esc_data[i]['temp']):
                    temps.append(esc_data[i]['temp'][j])
        
        if volts and total_power[j] > min_total_current * 40:  # Filter low power
            avg_volt = sum(volts) / len(volts)
            volt_power_pairs.append((avg_volt, total_power[j]))
            if temps:
                temp_for_pairs.append(sum(temps) / len(temps))
    
    if volt_power_pairs:
        vp_volts, vp_power = zip(*volt_power_pairs)
        if temp_for_pairs and len(temp_for_pairs) == len(vp_volts):
            scatter = ax.scatter(vp_volts, vp_power, c=temp_for_pairs, cmap='coolwarm', alpha=0.5, s=15, edgecolors='none')
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Temperature (°C)')
        else:
            ax.scatter(vp_volts, vp_power, color='blue', alpha=0.5, s=15, edgecolors='none')
        
        # Add trend line
        if len(vp_volts) > 10:
            z = np.polyfit(vp_volts, vp_power, 1)
            p = np.poly1d(z)
            v_range = np.linspace(min(vp_volts), max(vp_volts), 100)
            ax.plot(v_range, p(v_range), 'k-', linewidth=2, 
                   label=f'{z[0]:.1f} W/V')
            ax.legend(loc='upper right')
    
    ax.set_xlabel('Battery Voltage (V)')
    ax.set_ylabel('Total Power (W)')
    ax.set_title('Power Delivery vs Voltage')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_benchmark(esc_data, derived, active_escs, title_prefix, save_path):
    """Compare measured data against motor specification datasheet.
    
    Creates 2 subplots:
    1. Input Power vs RPM - compares measured RPM at given power to spec
    2. Efficiency (%) vs Input Power - compares motor efficiency curves
    """
    setup_style()
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{title_prefix} - Motor Benchmark vs {MOTOR_SPEC["name"]}', fontsize=14)
    
    # Extract spec data for curves
    spec_data = MOTOR_SPEC['data']
    spec_power = [spec_data[t][2] for t in sorted(spec_data.keys())]  # Input Power
    spec_rpm = [spec_data[t][4] for t in sorted(spec_data.keys())]    # RPM
    spec_eff = [spec_data[t][5] for t in sorted(spec_data.keys())]    # Efficiency %
    
    # Create smooth polynomial curves from spec data
    power_range = np.linspace(min(spec_power), max(spec_power), 200)
    
    # Fit polynomials (3rd order works well for these curves)
    rpm_poly = np.polyfit(spec_power, spec_rpm, 3)
    rpm_curve = np.poly1d(rpm_poly)(power_range)
    
    eff_poly = np.polyfit(spec_power, spec_eff, 3)
    eff_curve = np.poly1d(eff_poly)(power_range)
    
    # --- Subplot 1: Input Power vs RPM ---
    ax = axs[0]
    
    # Plot ±10% and ±20% error bands around spec curve
    ax.fill_between(power_range, rpm_curve * 0.8, rpm_curve * 1.2, 
                    color='gray', alpha=0.15, label='±20%')
    ax.fill_between(power_range, rpm_curve * 0.9, rpm_curve * 1.1, 
                    color='gray', alpha=0.25, label='±10%')
    
    # Plot spec curve
    ax.plot(power_range, rpm_curve, 'k-', linewidth=2.5, label=f'Spec: {MOTOR_SPEC["name"]}', alpha=0.8)
    ax.scatter(spec_power, spec_rpm, color='black', s=40, zorder=5, marker='s', label='Spec data points')
    
    # Plot measured data for each ESC
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            power = derived['per_esc'][i]['power']
            rpm = esc_data[i]['rpm']
            
            # Filter for reliable readings (above current threshold)
            curr = esc_data[i]['curr']
            valid_power = [p for p, c in zip(power, curr) if c >= MIN_CURRENT_THRESHOLD]
            valid_rpm = [r for r, c in zip(rpm, curr) if c >= MIN_CURRENT_THRESHOLD]
            
            if valid_power:
                ax.scatter(valid_power, valid_rpm, color=COLORS[i % 4], alpha=0.4, s=8, 
                          label=f'ESC {i} measured')
    
    ax.set_xlabel('Input Power (W)')
    ax.set_ylabel('RPM')
    ax.set_title('RPM vs Input Power')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)  # Start RPM axis at 0
    
    # --- Subplot 2: Efficiency vs Input Power ---
    ax = axs[1]
    
    # Plot spec efficiency curve
    ax.plot(power_range, eff_curve, 'k-', linewidth=2.5, label=f'Spec: {MOTOR_SPEC["name"]}', alpha=0.8)
    ax.scatter(spec_power, spec_eff, color='black', s=40, zorder=5, marker='s', label='Spec data points')
    
    # Plot measured efficiency (Output Power / Input Power * 100)
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            power = derived['per_esc'][i]['power']
            effs = derived['per_esc'][i]['efficiency']  # Now in % (output/input * 100)
            curr = esc_data[i]['curr']
            
            # Filter for reliable readings
            valid_power = []
            valid_eff = []
            for p, e, c in zip(power, effs, curr):
                if c >= MIN_CURRENT_THRESHOLD and e is not None and p > 10:
                    valid_power.append(p)
                    valid_eff.append(e)  # Already in %
            
            if valid_power:
                ax.scatter(valid_power, valid_eff, color=COLORS[i % 4], alpha=0.4, s=8,
                          label=f'ESC {i} measured')
    
    ax.set_xlabel('Input Power (W)')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Motor Efficiency vs Input Power')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)  # Allow slightly over 100% for calibration differences
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def export_csv(esc_data, derived, filepath):
    """Export current run data to CSV."""
    csv_path = filepath + "_export.csv"
    
    if not esc_data:
        print("No data to export.")
        return
    
    ref_inst = list(esc_data.keys())[0]
    n = len(esc_data[ref_inst]['time'])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['Time(s)']
        for i in sorted(esc_data.keys()):
            header.extend([f'ESC{i}_RPM', f'ESC{i}_Volt', f'ESC{i}_Curr', f'ESC{i}_Temp', f'ESC{i}_Power', f'ESC{i}_Eff'])
        header.extend(['Total_Curr', 'Total_Power'])
        writer.writerow(header)
        
        # Rows
        for j in range(n):
            row = [esc_data[ref_inst]['time'][j]]
            for i in sorted(esc_data.keys()):
                d = esc_data[i]
                deriv = derived['per_esc'].get(i, {})
                row.append(d['rpm'][j] if j < len(d['rpm']) else '')
                row.append(d['volt'][j] if j < len(d['volt']) else '')
                row.append(d['curr'][j] if j < len(d['curr']) else '')
                row.append(d['temp'][j] if j < len(d['temp']) else '')
                row.append(deriv.get('power', [])[j] if j < len(deriv.get('power', [])) else '')
                row.append(deriv.get('efficiency', [])[j] if j < len(deriv.get('efficiency', [])) else '')
            row.append(derived['total']['curr'][j] if j < len(derived['total']['curr']) else '')
            row.append(derived['total']['power'][j] if j < len(derived['total']['power']) else '')
            writer.writerow(row)
    
    print(f"Exported to: {csv_path}")


# =============================================================================
# Interactive Menu
# =============================================================================

def print_run_table(stats):
    print("\n" + "="*85)
    print(f"{'Run':<5} | {'Duration':<10} | {'Max RPM':<9} | {'Max Curr':<9} | {'Max Temp':<9} | {'Energy':<8} | {'Avg Eff':<8}")
    print("-"*85)
    for s in stats:
        print(f"{s['id']:<5} | {s['duration']:<10.1f} | {s['max_rpm']:<9.0f} | {s['max_curr']:<9.1f} | {s['max_temp']:<9.1f} | {s['energy_wh']:<8.2f} | {s['avg_eff']:<8.1f}")
    print("="*85)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_esc_data.py <bin_file>")
        sys.exit(1)
        
    filepath = sys.argv[1].strip().strip('"').strip("'")
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    print(f"\n{'='*50}")
    print(f"ESC Analysis Tool")
    print(f"File: {os.path.basename(filepath)}")
    print(f"{'='*50}")
    
    # Try to load from cache first
    all_esc_data = None
    runs = None
    
    if is_cache_valid(filepath):
        all_esc_data, runs = load_from_cache(filepath)
    
    # If cache miss or invalid, parse the full file
    if all_esc_data is None:
        print("\nParsing bin file (first time, this may take a moment)...")
        
        # Detect runs
        print("Scanning for runs...")
        runs = detect_runs(filepath)
        
        if not runs:
            print("No runs detected.")
            runs = [(0, 0)]  # Placeholder for full log
        
        # Load ALL ESC data from full file
        print("Loading all ESC data...")
        all_esc_data = load_esc_data(filepath, 0, 0)
        
        # Save to cache for next time
        save_to_cache(filepath, all_esc_data, runs)
    
    # Compute run stats (using cached data - FAST)
    stats = []
    if runs and runs[0] != (0, 0):
        print("Computing run stats...")
        stats = analyze_runs_from_cache(runs, all_esc_data)
        print_run_table(stats)
    else:
        print("No runs detected. Will analyze full log.")
    
    # State
    current_run_idx = None
    active_escs = []  # Will be auto-detected from data
    detected_esc_count = 0  # Track how many ESCs are in the log
    esc_data = {}  # Current run's data (filtered from all_esc_data)
    derived = {}
    filter_rc = DEFAULT_FILTER_RC  # Low-pass filter RC constant (adjustable)
    
    def filter_data_for_run(start_us, end_us):
        """Filter all_esc_data to just the specified time range."""
        filtered = defaultdict(lambda: {'time_us': [], 'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': []})
        
        for inst, data in all_esc_data.items():
            first_time = data['time_us'][0] if data['time_us'] else 0
            
            for i, t_us in enumerate(data['time_us']):
                if start_us and t_us < start_us:
                    continue
                if end_us and t_us > end_us:
                    break
                
                # Recalculate relative time from run start
                run_start = start_us if start_us else first_time
                t_sec = (t_us - run_start) / 1e6
                
                filtered[inst]['time_us'].append(t_us)
                filtered[inst]['time'].append(t_sec)
                filtered[inst]['rpm'].append(data['rpm'][i])
                filtered[inst]['volt'].append(data['volt'][i])
                filtered[inst]['curr'].append(data['curr'][i])
                filtered[inst]['temp'].append(data['temp'][i])
        
        return filtered
    
    def load_run(idx):
        nonlocal esc_data, derived, current_run_idx, active_escs, detected_esc_count, filter_rc
        
        if idx == 'combined':
            # Combined runs mode - stitch all runs together, filtering <10A data
            print(f"\nCombining all runs with current >= {MIN_CURRENT_THRESHOLD}A filter...")
            current_run_idx = 'combined'
            esc_data.clear()
            
            combined_data, _ = combine_runs_data(runs, all_esc_data)
            esc_data.update(combined_data)
            
        elif idx is None or idx == 'all':
            current_run_idx = 'all'
            print(f"\nFiltering data for full log...")
            esc_data.clear()
            esc_data.update(filter_data_for_run(0, 0))
            
        else:
            start, end = runs[idx]
            current_run_idx = idx
            print(f"\nFiltering data for run {idx + 1}...")
            esc_data.clear()
            esc_data.update(filter_data_for_run(start, end))
        
        # Compute derived metrics for whatever data was loaded
        derived.clear()
        derived.update(compute_derived_metrics(esc_data, filter_rc))
        
        # Auto-detect ESCs from loaded data
        detected_escs = sorted(esc_data.keys())
        detected_esc_count = len(detected_escs)
        active_escs = detected_escs.copy()
        
        print(f"Loaded {sum(len(d['time']) for d in esc_data.values())} data points from {detected_esc_count} ESCs: {detected_escs}")
    
    # Initial load prompt
    if len(runs) > 1 or (runs[0] != (0,0)):
        print("\nSelect a run to analyze:")
        print("  'a' = Full log (all data)")
        print(f"  'c' = All runs combined (filtered >{int(MIN_CURRENT_THRESHOLD)}A, no gaps)")
        choice = input("Run # (or 'a'/'c'/'q'): ").strip().lower()
        if choice == 'q':
            return
        elif choice == 'a':
            load_run('all')
        elif choice == 'c':
            load_run('combined')
        else:
            try:
                load_run(int(choice) - 1)
            except:
                print("Invalid. Loading all.")
                load_run('all')
    else:
        load_run('all')
    
    # Main menu loop
    while True:
        if isinstance(current_run_idx, int):
            run_label = f"Run #{current_run_idx + 1}"
        elif current_run_idx == 'combined':
            run_label = f"All Runs Combined (>{int(MIN_CURRENT_THRESHOLD)}A)"
        else:
            run_label = "Full Log"
        esc_label = ', '.join(str(e) for e in active_escs) if len(active_escs) < 4 else "All"
        filter_label = f"{filter_rc:.1f}s" if filter_rc > 0 else "Off"
        
        print(f"\n{'='*55}")
        print(f"Current: {run_label} | ESC: [{esc_label}] | Filter: {filter_label}")
        print(f"{'='*55}")
        print("[1] Plot ESC Basics (RPM, Volt, Curr, Temp)")
        print("[2] Plot Power (Total Current & Power)")
        print("[3] Plot Efficiency (RPM/Watt)")
        print("[4] Voltage Sag Analysis")
        print(f"[5] Benchmark vs {MOTOR_SPEC['name']}")
        print("[6] System Analysis (Voltage-Efficiency)")
        print("[7] Change Run")
        print("[8] Filter ESCs")
        print("[9] Export to CSV")
        print(f"[0] Adjust Low-Pass Filter (RC: {filter_label})")
        print("[q] Quit")
        
        choice = input("> ").strip().lower()
        
        title = f"{os.path.basename(filepath)} - {run_label}"
        # Save plots to organized output folder
        output_dir = get_output_dir(filepath)
        # Sanitize run_label for filename (remove invalid Windows chars)
        run_prefix = run_label.replace(' ', '_').replace('#', '').replace('>', 'gt').replace('<', 'lt').replace('(', '').replace(')', '')
        base_save = os.path.join(output_dir, run_prefix)
        
        if choice == '1':
            plot_esc_basics(esc_data, active_escs, title, f"{base_save}_basics.png", derived)
        elif choice == '2':
            plot_power(esc_data, derived, active_escs, title, f"{base_save}_power.png")
        elif choice == '3':
            plot_efficiency(esc_data, derived, active_escs, title, f"{base_save}_efficiency.png")
        elif choice == '4':
            plot_voltage_sag(esc_data, derived, active_escs, title, f"{base_save}_vsag.png", detected_esc_count)
        elif choice == '5':
            plot_benchmark(esc_data, derived, active_escs, title, f"{base_save}_benchmark.png")
        elif choice == '6':
            plot_system_analysis(esc_data, derived, active_escs, title, f"{base_save}_sysanalysis.png", detected_esc_count)
        elif choice == '7':
            # Change Run
            print_run_table(stats) if runs[0] != (0,0) else print("No runs to select.")
            print(f"  'a' = Full log | 'c' = All runs combined (filtered >{int(MIN_CURRENT_THRESHOLD)}A)")
            new_run = input("Run # (or 'a'/'c'): ").strip().lower()
            if new_run == 'a':
                load_run('all')
            elif new_run == 'c':
                load_run('combined')
            else:
                try:
                    load_run(int(new_run) - 1)
                except:
                    print("Invalid selection.")
        elif choice == '8':
            detected_list = sorted(esc_data.keys())
            print(f"Detected ESCs: {detected_list}")
            print(f"Current filter: {active_escs}")
            print("Enter ESC numbers to show (e.g., '0 2' or 'all'):")
            esc_input = input("> ").strip().lower()
            if esc_input == 'all':
                active_escs = detected_list.copy()
            else:
                try:
                    active_escs = [int(x) for x in esc_input.split()]
                    active_escs = [e for e in active_escs if e in esc_data]
                except:
                    print("Invalid. Keeping current filter.")
        elif choice == '9':
            export_csv(esc_data, derived, filepath)
        elif choice == '0':
            # Adjust low-pass filter
            print(f"\nLow-Pass Filter Settings:")
            print(f"  Current RC: {filter_rc:.2f}s" if filter_rc > 0 else "  Currently: Disabled")
            print(f"  Range: 0.25 - 5.0 seconds (higher = more smoothing)")
            print(f"  Enter 0 to disable filtering")
            try:
                new_rc = float(input("New RC value (seconds): ").strip())
                if new_rc < 0:
                    new_rc = 0
                elif new_rc > 0 and new_rc < 0.25:
                    new_rc = 0.25
                elif new_rc > 5.0:
                    new_rc = 5.0
                filter_rc = new_rc
                # Recompute derived metrics with new filter
                print(f"Applying filter RC={filter_rc:.2f}s..." if filter_rc > 0 else "Disabling filter...")
                derived.clear()
                derived.update(compute_derived_metrics(esc_data, filter_rc))
                print("Done! Metrics recomputed with new filter.")
            except:
                print("Invalid input. Keeping current filter.")
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
