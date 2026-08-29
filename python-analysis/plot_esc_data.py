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
MIN_CURRENT_THRESHOLD = 15.0  # Amps; ESC current telemetry is unreliable below this level

# Minimum throttle threshold for active motor data
# Data below this PWM value indicates motor is ramping down/stopped
MIN_THROTTLE_THRESHOLD = 1400  # PWM value (typically 1000-2000 range)

# Low-pass filter RC time constant (seconds)
# Higher = more smoothing, lower = less smoothing
# Range: 0.25 to 5.0 seconds (0 = disabled)
DEFAULT_FILTER_RC = 2.0

# Cache directory (stored next to the bin file)
CACHE_VERSION = "v9"  # v9 separates explicit RPM scaling from motor pole-pair metadata


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

# Built-in specs. Keys are used by --motor or config "motor_spec".
MOTOR_SPECS = {
    # MAD M6C10 EEE 200KV with FLUXER PRO 20x6.0 MATT prop, 12S.
    # The published MAD comparison table used an AMPX 60A ESC. The installed
    # AMPX 40A ESC should be treated as the current limit; the motor/prop load
    # and RPM comparison remain the applicable reference.
    "mad_m6c10_200kv_12s": {
        'name': 'MAD M6C10 EEE 200KV (12S)',
        'prop': 'FLUXER PRO 20x6.0 MATT',
        'note': 'MAD reference table uses AMPX 60A; installed configuration uses AMPX 40A',
        'data': {
            # Throttle %: [Voltage, Current, Input Power, Output Power, RPM, Efficiency %]
            30:  [48.24,  2.05,   98.5,   65.9, 2675, 66.92],
            35:  [48.24,  2.86,  137.5,   94.6, 3021, 68.77],
            40:  [48.25,  3.89,  187.1,  137.6, 3416, 73.49],
            45:  [48.23,  5.37,  258.4,  198.4, 3864, 76.76],
            50:  [48.16,  7.22,  347.2,  275.3, 4289, 79.25],
            55:  [48.16,  9.12,  438.5,  354.0, 4656, 80.68],
            60:  [48.16, 11.23,  540.1,  438.1, 4980, 81.06],
            65:  [48.15, 13.58,  653.7,  525.4, 5308, 80.33],
            70:  [48.13, 16.61,  798.9,  614.5, 5605, 77.13],
            75:  [48.04, 18.60,  892.9,  732.7, 5915, 82.05],
            80:  [48.03, 21.17, 1016.3,  848.3, 6234, 83.42],
            85:  [48.04, 26.12, 1254.5,  994.7, 6553, 79.24],
            90:  [47.97, 29.73, 1425.6, 1150.9, 6834, 80.75],
            95:  [47.93, 32.88, 1575.1, 1309.8, 7127, 83.14],
            100: [47.92, 39.79, 1906.6, 1505.3, 7470, 78.80],
        }
    },
    # MAD V62 PRO IPE 210KV with CF FLUXER 22.1x7.4 VTOL prop, AMPX 80A ESC, 12S
    # Source: Manufacturer datasheet
    "mad_v62_12s": {
        'name': 'MAD V62 PRO IPE 210KV (12S)',
        'prop': 'CF FLUXER 22.1x7.4 VTOL',
        'note': 'Spec at ~48V nominal',
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
    },
    # MAD V122 IPE 45KV with CB2 42x14 MATT prop, AMPX 200A ESC (12-24S), 24S spec
    "mad_v122_45kv_24s": {
        'name': 'MAD V122 IPE 45KV (24S reference)',
        'prop': 'CB2 42x14 MATT',
        'note': 'Spec at ~96V nominal (24S table)',
        'data': {
            # Throttle %: [Voltage, Current, Input Power, Output Power, RPM, Efficiency %]
            30:  [98.28, 3.81,   374.4,  277.8, 1310, 74.2],
            35:  [98.29, 5.59,   549.4,  427.9, 1510, 77.9],
            40:  [98.26, 7.53,   739.9,  598.5, 1689, 80.9],
            45:  [98.27, 10.00,  982.7,  808.3, 1877, 82.3],
            50:  [98.24, 13.32, 1308.6, 1110.2, 2078, 84.8],
            55:  [98.20, 18.28, 1795.1, 1520.2, 2301, 84.7],
            60:  [98.13, 23.86, 2341.4, 2030.6, 2517, 86.7],
            65:  [98.10, 30.33, 2975.4, 2562.2, 2712, 86.1],
            70:  [98.07, 35.35, 3466.8, 3105.8, 2898, 89.6],
            75:  [97.99, 44.72, 4382.1, 3828.2, 3080, 87.4],
            80:  [97.90, 54.94, 5378.6, 4679.4, 3271, 87.0],
            85:  [97.80, 66.30, 6484.1, 5595.8, 3455, 86.3],
            90:  [97.74, 77.16, 7541.6, 6350.0, 3639, 84.2],
            95:  [97.63, 90.73, 8858.0, 7245.8, 3812, 81.8],
            100: [97.43, 111.62,10875.1, 8493.5, 4029, 78.1],
        }
    }
}

DEFAULT_MOTOR_SPEC_KEY = "mad_m6c10_200kv_12s"

def get_motor_spec(spec_key):
    """Return a motor spec dict by key, or exit with an error."""
    if spec_key in MOTOR_SPECS:
        return MOTOR_SPECS[spec_key]
    print(f"Error: Unknown motor spec '{spec_key}'. Available: {', '.join(sorted(MOTOR_SPECS.keys()))}")
    sys.exit(1)

# Calculate propeller constant k from datasheet: P_out = k × RPM³
# For propellers: Output Power is proportional to RPM cubed
def calculate_propeller_constant(spec):
    """Derive propeller constant k from datasheet where P_out = k × RPM³."""
    k_values = []
    for _, data in spec['data'].items():
        output_power = data[3]  # Output Power (W)
        rpm = data[4]           # RPM
        if rpm > 0:
            k = output_power / (rpm ** 3)
            k_values.append(k)
    return sum(k_values) / len(k_values) if k_values else 0

def estimate_output_power(rpm, prop_k):
    """Estimate propeller output power from RPM using cubic relationship."""
    if rpm <= 0:
        return 0
    return prop_k * (rpm ** 3)

def calculate_motor_efficiency(rpm, input_power, prop_k):
    """Calculate motor efficiency as Output Power / Input Power.
    
    Uses the propeller cubic relationship to estimate output power from RPM.
    Returns efficiency as percentage, or None if invalid.
    """
    if input_power <= 0 or rpm <= 0:
        return None
    output_power = estimate_output_power(rpm, prop_k)
    efficiency = (output_power / input_power) * 100
    return efficiency


def load_json_config(config_path):
    """Load a JSON config file for analysis settings."""
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)


def normalize_esc_channel_map(value):
    """Normalize ESC channel map to a dict of int ESC -> int channel (1-8)."""
    if value is None:
        return None
    if isinstance(value, list):
        return {i: int(v) for i, v in enumerate(value)}
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    return None


def esc_channel_map_to_meta(esc_channel_map):
    """Convert ESC channel map to JSON-friendly dict with string keys."""
    if not esc_channel_map:
        return None
    return {str(k): int(v) for k, v in esc_channel_map.items()}


def normalize_current_scale_rules(rules, pwm_min=1000, pwm_max=2000):
    """Normalize current scale rules with PWM thresholds.
    
    Rules can use min/max_throttle_pwm or min/max_throttle_pct.
    """
    if not rules:
        return []
    normalized = []
    for rule in rules:
        try:
            scale = float(rule.get('scale', 1.0))
        except Exception:
            continue

        min_pwm = rule.get('min_throttle_pwm')
        max_pwm = rule.get('max_throttle_pwm')
        min_pct = rule.get('min_throttle_pct')
        max_pct = rule.get('max_throttle_pct')

        if min_pwm is None and min_pct is not None:
            min_pwm = pwm_min + (pwm_max - pwm_min) * (float(min_pct) / 100.0)
        if max_pwm is None and max_pct is not None:
            max_pwm = pwm_min + (pwm_max - pwm_min) * (float(max_pct) / 100.0)

        normalized.append({
            'scale': scale,
            'min_pwm': float(min_pwm) if min_pwm is not None else None,
            'max_pwm': float(max_pwm) if max_pwm is not None else None
        })
    return normalized


def get_current_scale(max_throttle_pwm, rules):
    """Return the scale factor for a given max throttle PWM."""
    if not rules:
        return 1.0
    scale = 1.0
    for rule in rules:
        min_pwm = rule.get('min_pwm')
        max_pwm = rule.get('max_pwm')
        if min_pwm is not None and max_throttle_pwm < min_pwm:
            continue
        if max_pwm is not None and max_throttle_pwm > max_pwm:
            continue
        scale = rule.get('scale', 1.0)
    return scale


def describe_current_scale_rules(rules):
    """Human-readable summary of current scale rules."""
    if not rules:
        return "None"
    parts = []
    for rule in rules:
        conds = []
        if rule.get('min_pwm') is not None:
            conds.append(f">={rule['min_pwm']:.0f}")
        if rule.get('max_pwm') is not None:
            conds.append(f"<={rule['max_pwm']:.0f}")
        cond = " & ".join(conds) if conds else "any"
        parts.append(f"{cond}: x{rule.get('scale', 1.0):g}")
    return "; ".join(parts)


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

def get_poles_cache_key(poles, rpm_scale=1.0):
    """Generate a filename suffix for pole metadata and explicit RPM scale."""
    if isinstance(poles, int):
        pole_key = f"p{poles}"
    elif isinstance(poles, str):
        # Should be int if simple, but handle str just in case
        pole_key = f"p{poles}"
    elif isinstance(poles, dict):
        # Create a deterministic string for the dict
        # e.g. p_mixed_HASH
        import hashlib
        # Sort items to ensure stability
        s = json.dumps(dict(sorted(poles.items())), sort_keys=True)
        h = hashlib.md5(s.encode()).hexdigest()[:8]
        pole_key = f"p_mixed_{h}"
    else:
        pole_key = "p_unknown"
    scale_key = f"s{float(rpm_scale):g}".replace('-', 'm').replace('.', 'p')
    return f"{pole_key}_{scale_key}"

def get_cache_path(filepath, poles, rpm_scale=1.0):
    """Get the cache file path for a pole description and RPM scale."""
    output_dir = get_output_dir(filepath)
    suffix = get_poles_cache_key(poles, rpm_scale)
    return os.path.join(output_dir, f"esc_data_cache_{suffix}.csv")

def get_cache_meta_path(filepath, poles, rpm_scale=1.0):
    """Get the cache metadata file path."""
    output_dir = get_output_dir(filepath)
    suffix = get_poles_cache_key(poles, rpm_scale)
    return os.path.join(output_dir, f"cache_meta_{suffix}.json")

def is_cache_valid(filepath, poles, rpm_scale, esc_channel_map):
    """Check if cached data exists and is still valid."""
    cache_path = get_cache_path(filepath, poles, rpm_scale)
    meta_path = get_cache_meta_path(filepath, poles, rpm_scale)
    
    if not os.path.exists(cache_path) or not os.path.exists(meta_path):
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # Check version
        if meta.get('version') != CACHE_VERSION:
            print("Cache version mismatch, reparsing...")
            return False
            
        # Check pole count matches
        if meta.get('poles') != poles:
            print(f"Cache pole count mismatch ({meta.get('poles')} vs {poles}), reparsing...")
            return False

        if float(meta.get('rpm_scale', 1.0)) != float(rpm_scale):
            print(f"Cache RPM scale mismatch ({meta.get('rpm_scale')} vs {rpm_scale}), reparsing...")
            return False

        # Check ESC channel map matches
        if meta.get('esc_channel_map') != esc_channel_map_to_meta(esc_channel_map):
            print("Cache ESC channel map mismatch, reparsing...")
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


def load_from_cache(filepath, poles, rpm_scale):
    """Load ESC data and runs from cache (CSV format)."""
    cache_path = get_cache_path(filepath, poles, rpm_scale)
    meta_path = get_cache_meta_path(filepath, poles, rpm_scale)
    
    print(f"Loading from cache (Poles: {poles}, RPM scale: {rpm_scale:g})...")
    
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
            inst_key = int(inst)
            esc_data[inst_key] = {
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


def save_to_cache(filepath, esc_data, runs, poles, rpm_scale, esc_channel_map):
    """Save ESC data and runs to cache (CSV format for easy viewing)."""
    cache_path = get_cache_path(filepath, poles, rpm_scale)
    meta_path = get_cache_meta_path(filepath, poles, rpm_scale)
    
    print(f"Saving to cache (Poles: {poles}, RPM scale: {rpm_scale:g})...")
    
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
            'esc_count': len(esc_data),
            'poles': poles,
            'rpm_scale': float(rpm_scale),
            'esc_channel_map': esc_channel_map_to_meta(esc_channel_map)
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


def load_esc_data(filepath, start_us=0, end_us=0, poles=14, rpm_scale=1.0, esc_channel_map=None):
    """Load ESC data within optional time range. Returns dict by instance.
    
    Args:
        filepath: Path to .BIN file
        start_us, end_us: Time range (0=all)
        poles: Motor pole-pair metadata (does not alter RPM)
        rpm_scale: Explicit multiplier applied directly to logged RPM (default 1.0)
        esc_channel_map: Optional dict mapping ESC instance -> RCOU channel (1-8)
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
        """Find closest throttle value for given timestamp.
        
        Uses per-ESC channel map if provided; otherwise uses max throttle.
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
        
        # Use per-ESC channel mapping if provided, else max throttle
        if esc_channel_map and esc_instance in esc_channel_map:
            ch = esc_channel_map.get(esc_instance)
            if ch is not None and 1 <= int(ch) <= 8:
                return throttle_lookup.get(t, {}).get(int(ch) - 1, 0)
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
            # Pole-pair count is descriptive only. Logged RPM is trusted as-is
            # unless the user supplies an explicit --rpm-scale multiplier.
            esc_data[i]['rpm'].append(msg.RPM * float(rpm_scale))
            
            esc_data[i]['volt'].append(msg.Volt)
            esc_data[i]['curr'].append(msg.Curr)
            esc_data[i]['temp'].append(msg.Temp)
            esc_data[i]['throttle'].append(get_throttle(msg.TimeUS, i))
        except:
            continue
            
    return esc_data


def _interpolate_without_bridging_gaps(times, values, grid):
    """Interpolate one ESC series while leaving startup/dropout periods as NaN."""
    if len(times) < 2 or len(values) < 2 or len(grid) == 0:
        return np.full(len(grid), np.nan)
    n = min(len(times), len(values))
    t = np.asarray(times[:n], dtype=float)
    y = np.asarray(values[:n], dtype=float)
    valid = np.isfinite(t) & np.isfinite(y)
    t, y = t[valid], y[valid]
    if len(t) < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(t)
    t, y = t[order], y[order]
    t, unique_idx = np.unique(t, return_index=True)
    y = y[unique_idx]
    if len(t) < 2:
        return np.full(len(grid), np.nan)

    result = np.interp(grid, t, y, left=np.nan, right=np.nan)
    sample_dt = np.diff(t)
    typical_dt = float(np.median(sample_dt[sample_dt > 0])) if np.any(sample_dt > 0) else 0.25
    max_nearest_distance = max(0.5, 2.5 * typical_dt)
    right_idx = np.searchsorted(t, grid, side='left')
    left_idx = np.clip(right_idx - 1, 0, len(t) - 1)
    right_idx = np.clip(right_idx, 0, len(t) - 1)
    nearest = np.minimum(np.abs(grid - t[left_idx]), np.abs(t[right_idx] - grid))
    result[nearest > max_nearest_distance] = np.nan
    return result


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
            if i >= MIN_CURRENT_THRESHOLD and p > 1.0:
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


def combine_runs_data(runs, all_esc_data, current_threshold=None, throttle_threshold=MIN_THROTTLE_THRESHOLD):
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
        current_threshold = MIN_CURRENT_THRESHOLD

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
    
    synchronized = build_synchronized_total_series(esc_data, derived['per_esc'], active_escs)
    ref_time = synchronized['time']
    
    # Get active time range to crop out low-power startup/shutdown
    t_start, t_end = get_active_time_range(esc_data)
    
    # Total Current
    ax = axs[0]
    total_curr = synchronized['curr']
    ax.plot(ref_time, total_curr, color='black', linewidth=2, label='Total')
    ax.set_ylabel('Total Current (A)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    finite_total_curr = [x for x in total_curr if np.isfinite(x)]
    if finite_total_curr:
        stats = f"Min: {min(finite_total_curr):.1f}  Med: {np.median(finite_total_curr):.1f}  Max: {max(finite_total_curr):.1f}"
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
    total_power = synchronized['power']
    ax.plot(ref_time, total_power, color='purple', linewidth=2, label='Total Power')
    ax.set_ylabel('Total Power (W)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    finite_total_power = [x for x in total_power if np.isfinite(x)]
    if finite_total_power:
        stats = f"Min: {min(finite_total_power):.1f}  Med: {np.median(finite_total_power):.1f}  Max: {max(finite_total_power):.1f}"
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


def plot_efficiency(esc_data, derived, active_escs, title_prefix, save_path, mode='pct'):
    """Plot Efficiency for selected ESCs. Only shows valid readings.
    
    Args:
        mode: 'pct' for Motor Efficiency (%), 'rpm_w' for RPM/Watt
    """
    setup_style()
    
    ylabel = 'Efficiency (%)' if mode == 'pct' else 'Efficiency (RPM/W)'
    data_key = 'efficiency_pct' if mode == 'pct' else 'efficiency_rpm_w'
    
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle(f'{title_prefix} - {ylabel} [Current >= {MIN_CURRENT_THRESHOLD}A]', fontsize=14)
    
    # Get active time range for cropping
    t_start, t_end = get_active_time_range(esc_data)
    
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            times = esc_data[i]['time']
            # Fallback for old cache or safety
            if data_key not in derived['per_esc'][i]:
                effs = derived['per_esc'][i]['efficiency']
            else:
                effs = derived['per_esc'][i][data_key]
            
            # Filter out None values (unreliable low-current readings)
            valid_t = [t for t, e in zip(times, effs) if e is not None]
            valid_e = [e for e in effs if e is not None]
            if valid_t:
                label = f'ESC {i} ({min(valid_e):.1f}/{np.median(valid_e):.1f}/{max(valid_e):.1f})'
                ax.plot(valid_t, valid_e,
                       label=label, color=COLORS[i % 4], linewidth=1.2, alpha=0.85)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    if mode == 'pct':
        ax.set_ylim(0, 110)
    
    # Crop X-axis to active range
    if t_start is not None and t_end is not None:
        ax.set_xlim(t_start, t_end)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved: {save_path}")
    plt.show()


def plot_voltage_sag(esc_data, derived, active_escs, title_prefix, save_path, esc_count=4, esc_count_series=None):
    """Scatter plot of Voltage vs TOTAL Current, colored by avg Temperature.
    
    Uses total current (sum of all ESCs) since battery voltage sag is a function
    of total system current draw, not individual ESC current.
    
    Args:
        esc_count: Number of ESCs in the system (for threshold calculation)
    """
    min_total_current = MIN_CURRENT_THRESHOLD * esc_count
    if esc_count_series:
        valid_counts = [c for c in esc_count_series if c > 0]
        if valid_counts:
            min_count = min(valid_counts)
            max_count = max(valid_counts)
            count_label = f"{min_count}-{max_count}" if min_count != max_count else f"{min_count}"
            min_total_current = MIN_CURRENT_THRESHOLD * min_count
        else:
            count_label = f"{esc_count}"
    else:
        count_label = f"{esc_count}"
    
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'{title_prefix} - Voltage Sag Analysis [Total Current >= {min_total_current}A (ESCs: {count_label})]', fontsize=14)
    
    synchronized = build_synchronized_total_series(esc_data, derived['per_esc'], active_escs)
    total_curr = synchronized['curr']
    ref_times = synchronized['time']
    synchronized_counts = synchronized['esc_count']
    
    if not total_curr or not ref_times:
        ax.text(0.5, 0.5, 'No data available',
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
        plt.show()
        return
    
    avg_volt = []
    avg_temp = []
    valid_curr = []

    for j, curr in enumerate(total_curr):
        if not np.isfinite(curr):
            continue
        count = synchronized_counts[j] if j < len(synchronized_counts) else esc_count
        if count <= 0:
            continue
        min_total_current_j = MIN_CURRENT_THRESHOLD * count
        voltage = synchronized['avg_volt'][j]
        temperature = synchronized['avg_temp'][j]
        if curr >= min_total_current_j and np.isfinite(voltage):
            avg_volt.append(voltage)
            avg_temp.append(temperature if np.isfinite(temperature) else 0)
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


def plot_system_analysis(esc_data, derived, active_escs, title_prefix, save_path, esc_count, mode='pct', esc_count_series=None):
    """Comprehensive system analysis showing voltage, current, power, and efficiency relationships.
    
    Creates 4 subplots:
    1. Voltage vs Efficiency (based on mode) - colored by current
    2. Power vs Efficiency (based on mode) - shows efficiency drop at high power
    3. ESC Input Power vs Efficiency - per-ESC scatter and median curves
    4. Voltage vs Total Power - shows power delivery at different voltage levels
    
    Args:
        mode: 'pct' for Motor Efficiency (%), 'rpm_w' for RPM/Watt
    """
    min_total_current = MIN_CURRENT_THRESHOLD * esc_count
    count_label = f"{esc_count}"
    if esc_count_series:
        valid_counts = [c for c in esc_count_series if c > 0]
        if valid_counts:
            min_count = min(valid_counts)
            max_count = max(valid_counts)
            count_label = f"{min_count}-{max_count}" if min_count != max_count else f"{min_count}"
            min_total_current = MIN_CURRENT_THRESHOLD * min_count
    
    setup_style()
    
    ylabel = 'Efficiency (%)' if mode == 'pct' else 'Efficiency (RPM/W)'
    data_key = 'efficiency_pct' if mode == 'pct' else 'efficiency_rpm_w'
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'{title_prefix} - System Analysis [Current >= {min_total_current:.0f}A total (ESCs: {count_label})]', fontsize=14)
    
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
            curr_list = derived['per_esc'][i].get('curr_filtered', esc_data[i]['curr'])
            if j >= len(curr_list):
                continue
            curr = curr_list[j]
            if curr < MIN_CURRENT_THRESHOLD:
                continue
            
            volt = esc_data[i]['volt'][j]
            temp = esc_data[i]['temp'][j]
            power = derived['per_esc'][i]['power'][j] if j < len(derived['per_esc'][i]['power']) else 0
            
            # Use selected efficiency metric
            if data_key in derived['per_esc'][i]:
                eff_list = derived['per_esc'][i][data_key]
            else:
                eff_list = derived['per_esc'][i]['efficiency'] # Fallback
                
            eff = eff_list[j] if j < len(eff_list) else None
            
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
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Voltage')
    ax.grid(True, alpha=0.3)
    if mode == 'pct': ax.set_ylim(0, 110)
    
    # --- Panel 2: Power vs Efficiency ---
    ax = axs[0, 1]
    scatter = ax.scatter(all_power, all_eff, c=all_volts, cmap='coolwarm', alpha=0.5, s=15, edgecolors='none')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Voltage (V)')
    
    ax.set_xlabel('Input Power (W)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Power (colored by Voltage)')
    ax.grid(True, alpha=0.3)
    if mode == 'pct': ax.set_ylim(0, 110)
    
    # --- Panel 3: ESC Input Power vs Efficiency per ESC ---
    # Use common power bins for every ESC so the median curves can be compared
    # at the same electrical operating points. Raw samples remain visible to
    # show spread and telemetry artifacts.
    ax = axs[1, 0]
    power_bin_edges = np.linspace(min(all_power), max(all_power), 26)
    for i in active_escs:
        esc_power = np.asarray(per_esc_data_local[i]['power'], dtype=float)
        esc_eff = np.asarray(per_esc_data_local[i]['eff'], dtype=float)
        valid = np.isfinite(esc_power) & np.isfinite(esc_eff) & (esc_power > 0)
        esc_power = esc_power[valid]
        esc_eff = esc_eff[valid]
        if not len(esc_power):
            continue

        color = COLORS[i % 4]
        ax.scatter(
            esc_power, esc_eff, color=color, alpha=0.14, s=9,
            edgecolors='none'
        )

        bin_ids = np.digitize(esc_power, power_bin_edges) - 1
        median_power = []
        median_eff = []
        for bin_id in range(len(power_bin_edges) - 1):
            in_bin = bin_ids == bin_id
            # Suppress bins that only contain short transients; those sparse
            # high-power samples otherwise create misleading median spikes.
            if np.count_nonzero(in_bin) < 20:
                continue
            median_power.append(float(np.median(esc_power[in_bin])))
            median_eff.append(float(np.median(esc_eff[in_bin])))

        if median_power:
            ax.plot(
                median_power, median_eff, color=color, linewidth=2.0,
                marker='o', markersize=3, label=f'ESC {i} median'
            )
        else:
            ax.scatter([], [], color=color, s=20, label=f'ESC {i}')
    
    ax.set_xlabel('ESC Input Power (W = V × I)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs ESC Input Power (per ESC)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    if mode == 'pct': ax.set_ylim(0, 110)
    
    # --- Panel 4: Voltage vs Total Power ---
    ax = axs[1, 1]
    # Calculate total power for each voltage point
    synchronized = build_synchronized_total_series(esc_data, derived['per_esc'], active_escs)
    total_power = synchronized['power']
    ref_times = synchronized['time']
    
    volt_power_pairs = []
    # ... rest of function ...
    temp_for_pairs = []
    for j in range(min(len(ref_times), len(total_power))):
        avg_volt = synchronized['avg_volt'][j]
        avg_temp = synchronized['avg_temp'][j]
        active_count = synchronized['esc_count'][j]
        point_threshold = MIN_CURRENT_THRESHOLD * max(active_count, 1) * 40
        if np.isfinite(avg_volt) and np.isfinite(total_power[j]) and total_power[j] > point_threshold:
            volt_power_pairs.append((avg_volt, total_power[j]))
            temp_for_pairs.append(avg_temp if np.isfinite(avg_temp) else 0)
    
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


def plot_efficiency_power_time_3d(esc_data, derived, active_escs, title_prefix, save_path, mode='pct'):
    """Plot per-ESC efficiency against input power and time, colored by voltage.

    The graph window is rotatable. Separate ESC panels keep the voltage color
    scale available for operating-history comparisons without using color for
    both ESC identity and voltage.
    """
    setup_style()

    ylabel = 'Efficiency (%)' if mode == 'pct' else 'Efficiency (RPM/W)'
    data_key = 'efficiency_pct' if mode == 'pct' else 'efficiency_rpm_w'
    selected = [
        i for i in active_escs
        if i in esc_data and i in derived.get('per_esc', {})
    ]

    valid_by_esc = {}
    all_voltages = []
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
            & (currents >= MIN_CURRENT_THRESHOLD) & (powers > 0)
        )
        rows = {
            'time': times[valid], 'volt': volts[valid],
            'power': powers[valid], 'efficiency': efficiencies[valid],
        }
        valid_by_esc[i] = rows
        all_voltages.extend(rows['volt'].tolist())

    if not selected or not all_voltages:
        print('No valid data available for the 3D efficiency plot.')
        return

    voltage_min = float(np.min(all_voltages))
    voltage_max = float(np.max(all_voltages))
    if voltage_max <= voltage_min:
        voltage_max = voltage_min + 1.0

    column_count = 2 if len(selected) > 1 else 1
    row_count = int(np.ceil(len(selected) / column_count))
    fig = plt.figure(figsize=(14, max(6, 5.5 * row_count)))
    fig.suptitle(
        f'{title_prefix} - {ylabel} vs ESC Input Power and Time '
        f'[Current >= {MIN_CURRENT_THRESHOLD:g}A]',
        fontsize=14
    )
    axes = []
    first_scatter = None

    for panel, i in enumerate(selected, start=1):
        ax = fig.add_subplot(row_count, column_count, panel, projection='3d')
        axes.append(ax)
        rows = valid_by_esc[i]
        point_count = len(rows['time'])
        if point_count == 0:
            ax.text2D(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'ESC {i}')
            continue

        # Preserve the complete time span while keeping rotation responsive.
        stride = max(1, int(np.ceil(point_count / 4000)))
        take = slice(None, None, stride)
        scatter = ax.scatter(
            rows['power'][take], rows['time'][take], rows['efficiency'][take],
            c=rows['volt'][take], cmap='viridis',
            vmin=voltage_min, vmax=voltage_max,
            s=4, alpha=0.16, edgecolors='none', depthshade=False
        )
        if first_scatter is None:
            first_scatter = scatter

        # Trace the response in voltage slices so the shape is readable without
        # having to infer it from a dense 3D point cloud. Each trajectory uses
        # medians in power bins; time remains an actual measured coordinate.
        voltage_edges = np.linspace(voltage_min, voltage_max, 5)
        power_lo, power_hi = np.percentile(rows['power'], [1, 99])
        power_edges = np.linspace(power_lo, power_hi, 25)
        color_map = plt.get_cmap('viridis')
        for band_index in range(len(voltage_edges) - 1):
            lower, upper = voltage_edges[band_index:band_index + 2]
            if band_index == len(voltage_edges) - 2:
                in_voltage_band = (rows['volt'] >= lower) & (rows['volt'] <= upper)
            else:
                in_voltage_band = (rows['volt'] >= lower) & (rows['volt'] < upper)
            curve_power, curve_time, curve_eff = [], [], []
            for power_start, power_end in zip(power_edges[:-1], power_edges[1:]):
                in_bin = (
                    in_voltage_band & (rows['power'] >= power_start)
                    & (rows['power'] < power_end)
                )
                if np.count_nonzero(in_bin) < 20:
                    continue
                curve_power.append(float(np.median(rows['power'][in_bin])))
                curve_time.append(float(np.median(rows['time'][in_bin])))
                curve_eff.append(float(np.median(rows['efficiency'][in_bin])))
            if len(curve_power) >= 2:
                band_voltage = 0.5 * (lower + upper)
                normalized_voltage = (band_voltage - voltage_min) / (voltage_max - voltage_min)
                ax.plot(
                    curve_power, curve_time, curve_eff,
                    color=color_map(normalized_voltage), linewidth=2.8,
                    marker='o', markersize=2.8, alpha=0.95
                )
        ax.set_title(f'ESC {i}')
        ax.set_xlabel('Input Power (W)')
        ax.set_ylabel('Time (s)')
        ax.set_zlabel(ylabel)
        if mode == 'pct':
            ax.set_zlim(0, 110)
        ax.view_init(elev=23, azim=-58)
        ax.grid(True, alpha=0.25)

    fig.subplots_adjust(left=0.04, right=0.87, bottom=0.05, top=0.91, wspace=0.08, hspace=0.14)
    if first_scatter is not None:
        colorbar_axis = fig.add_axes([0.91, 0.18, 0.016, 0.64])
        colorbar = fig.colorbar(first_scatter, cax=colorbar_axis)
        colorbar.set_label('ESC Voltage (V)')
    fig.savefig(save_path, dpi=140)
    print(f'Saved rotatable 3D efficiency plot: {save_path}')
    print('Drag in the graph window to rotate; use the toolbar to zoom or reset the view.')
    plt.show()


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
            & (currents >= MIN_CURRENT_THRESHOLD) & (powers > 0)
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
        f'[Current >= {MIN_CURRENT_THRESHOLD:g}A]', fontsize=14
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


def plot_benchmark(esc_data, derived, active_escs, title_prefix, save_path, motor_spec, mode='pct'):
    """Compare measured data against motor specification datasheet.
    
    Creates 2 subplots:
    1. Input Power vs RPM - compares measured RPM at given power to spec
    2. Efficiency (based on mode) vs Input Power - compares curves
    
    Args:
        mode: 'pct' for Motor Efficiency (%), 'rpm_w' for RPM/Watt
    """
    setup_style()
    
    ylabel = 'Efficiency (%)' if mode == 'pct' else 'Efficiency (RPM/W)'
    data_key = 'efficiency_pct' if mode == 'pct' else 'efficiency_rpm_w'
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{title_prefix} - Motor Benchmark [Current >= {MIN_CURRENT_THRESHOLD}A]', fontsize=14)
    
    # Extract spec data for curves
    spec_data = motor_spec['data']
    spec_power = [spec_data[t][2] for t in sorted(spec_data.keys())]  # Input Power
    spec_rpm = [spec_data[t][4] for t in sorted(spec_data.keys())]    # RPM
    spec_eff = [spec_data[t][5] for t in sorted(spec_data.keys())]    # Efficiency %
    
    # Calculate RPM/W for spec if needed
    if mode == 'rpm_w':
        spec_eff_curve_data = [r/p if p > 0 else 0 for r, p in zip(spec_rpm, spec_power)]
    else:
        spec_eff_curve_data = spec_eff
    
    # Create smooth polynomial curves from spec data
    power_range = np.linspace(min(spec_power), max(spec_power), 200)
    
    # Fit polynomials (3rd order works well for these curves)
    rpm_poly = np.polyfit(spec_power, spec_rpm, 3)
    rpm_curve = np.poly1d(rpm_poly)(power_range)
    
    eff_poly = np.polyfit(spec_power, spec_eff_curve_data, 3)
    eff_curve = np.poly1d(eff_poly)(power_range)
    
    # --- Subplot 1: Input Power vs RPM ---
    ax = axs[0]
    
    # Plot ±10% and ±20% error bands around spec curve
    ax.fill_between(power_range, rpm_curve * 0.8, rpm_curve * 1.2, 
                    color='gray', alpha=0.15, label='±20%')
    ax.fill_between(power_range, rpm_curve * 0.9, rpm_curve * 1.1, 
                    color='gray', alpha=0.25, label='±10%')
    
    # Plot spec curve
    ax.plot(power_range, rpm_curve, 'k-', linewidth=2.5, label=f'Spec: {motor_spec["name"]}', alpha=0.8)
    ax.scatter(spec_power, spec_rpm, color='black', s=40, zorder=5, marker='s', label='Spec data points')
    
    # Plot measured data for each ESC
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            power = derived['per_esc'][i]['power']
            rpm = esc_data[i]['rpm']
            
            # Filter for reliable readings (above current threshold)
            curr = derived['per_esc'][i].get('curr_filtered', esc_data[i]['curr'])
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
    ax.plot(power_range, eff_curve, 'k-', linewidth=2.5, label=f'Spec: {motor_spec["name"]}', alpha=0.8)
    if mode == 'pct':
        ax.scatter(spec_power, spec_eff, color='black', s=40, zorder=5, marker='s', label='Spec data points')
    else:
        ax.scatter(spec_power, spec_eff_curve_data, color='black', s=40, zorder=5, marker='s', label='Spec data points')
    
    # Plot measured efficiency
    for i in active_escs:
        if i in esc_data and i in derived['per_esc']:
            power = derived['per_esc'][i]['power']
            
            # Fallback for old cache or safety
            if data_key not in derived['per_esc'][i]:
                effs = derived['per_esc'][i]['efficiency']
            else:
                effs = derived['per_esc'][i][data_key]
                
            curr = derived['per_esc'][i].get('curr_filtered', esc_data[i]['curr'])
            
            # Filter for reliable readings
            valid_power = []
            valid_eff = []
            for p, e, c in zip(power, effs, curr):
                if c >= MIN_CURRENT_THRESHOLD and e is not None and p > 10:
                    valid_power.append(p)
                    valid_eff.append(e)
            
            if valid_power:
                ax.scatter(valid_power, valid_eff, color=COLORS[i % 4], alpha=0.4, s=8,
                          label=f'ESC {i} measured')
    
    ax.set_xlabel('Input Power (W)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} vs Input Power')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    if mode == 'pct':
        ax.set_ylim(0, 110)
    
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
                curr_vals = deriv.get('curr_filtered', d.get('curr', []))
                row.append(curr_vals[j] if j < len(curr_vals) else '')
                row.append(d['temp'][j] if j < len(d['temp']) else '')
                row.append(deriv.get('power', [])[j] if j < len(deriv.get('power', [])) else '')
                row.append(deriv.get('efficiency', [])[j] if j < len(deriv.get('efficiency', [])) else '')
            row.append(derived['total']['curr'][j] if j < len(derived['total']['curr']) else '')
            row.append(derived['total']['power'][j] if j < len(derived['total']['power']) else '')
            writer.writerow(row)
    
    print(f"Exported to: {csv_path}")


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
    """Show voltage versus time and measure energy between two clicked times."""
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
    fig.suptitle(f"{title_prefix} - Voltage vs Time / Energy Interval", fontsize=14)
    curve_time = np.where(valid, time_s, np.nan)
    curve_voltage = np.where(valid, voltage, np.nan)
    ax.plot(curve_time, curve_voltage, color='0.45', linewidth=1.0, alpha=0.65)

    # Keep the scatter responsive on large logs while retaining the full curve
    # for nearest-point selection and energy calculations.
    plot_step = max(1, len(valid_indices) // 3500)
    plotted = valid_indices[::plot_step]
    scatter = ax.scatter(
        time_s[plotted], voltage[plotted], c=current[plotted],
        cmap='viridis', s=9, alpha=0.65, edgecolors='none'
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label('Summed ESC current (A)')
    ax.set_xlabel('Time since selected log start (s)')
    ax.set_ylabel('Average reporting-ESC voltage (V)')
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
                f"P1  {voltage[idx]:.3f} V | {cumulative_wh[idx]:.2f} Wh | "
                f"t={time_s[idx]:.1f}s | ESCs={esc_counts[idx]}"
            )
        elif len(selected_indices) == 2:
            first, second = sorted(selected_indices, key=lambda idx: time_s[idx])
            delta_v = voltage[second] - voltage[first]
            energy_used = cumulative_wh[second] - cumulative_wh[first]
            charge_used = cumulative_ah[second] - cumulative_ah[first]
            segment = slice(first, second + 1)
            highlight, = ax.plot(
                curve_time[segment], curve_voltage[segment],
                color=COLORS[2 % len(COLORS)], linewidth=2.8,
                alpha=0.9, zorder=6
            )
            selection_artists.append(highlight)
            result_text.set_text(
                f"V1={voltage[first]:.3f} V   V2={voltage[second]:.3f} V   "
                f"ΔV={delta_v:+.3f} V\n"
                f"Energy used={energy_used:.3f} Wh   Charge used={charge_used:.3f} Ah   "
                f"Δt={time_s[second] - time_s[first]:.1f}s"
            )
            fig.savefig(save_path, dpi=140)
            print(
                f"Selected {voltage[first]:.3f} V -> {voltage[second]:.3f} V: "
                f"{energy_used:.3f} Wh, {charge_used:.3f} Ah "
                f"(delta V {delta_v:+.3f} V)"
            )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax:
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


# =============================================================================
# Interactive Menu
# =============================================================================

def print_run_table(stats):
    print("\n" + "="*99)
    print(f"{'Run':<5} | {'Duration':<10} | {'Max RPM':<9} | {'Max/ESC A':<9} | {'Max Temp':<9} | {'ESC Wh':<9} | {'ESC Ah':<8} | {'Avg Eff':<8}")
    print("-"*99)
    for s in stats:
        print(f"{s['id']:<5} | {s['duration']:<10.1f} | {s['max_rpm']:<9.0f} | {s['max_curr']:<9.1f} | {s['max_temp']:<9.1f} | {s['energy_wh']:<9.2f} | {s.get('charge_ah', 0):<8.2f} | {s['avg_eff']:<8.1f}")
    print("="*99)


def main():
    global MIN_CURRENT_THRESHOLD

    import argparse
    parser = argparse.ArgumentParser(description='Analyze ArduPilot .BIN log for ESC data')
    parser.add_argument('filepath', nargs='?', help='Path to .BIN file')
    # Use nargs='+' to retain optional per-ESC pole metadata.
    parser.add_argument('--poles', nargs='+', default=["14"],
                        help='Motor pole-pair metadata (default: 14; does not scale logged RPM)')
    parser.add_argument('--rpm-scale', type=float, default=None,
                        help='Explicit multiplier for logged RPM (default: 1.0)')
    parser.add_argument('--min-current', type=float, default=None,
                        help='Per-ESC reliability/filter threshold in amps (default: 15)')
    parser.add_argument('--config', help='Path to JSON config file with analysis overrides')
    parser.add_argument('--motor', choices=sorted(MOTOR_SPECS.keys()),
                        help=f"Comparison model (default: {DEFAULT_MOTOR_SPEC_KEY})")
    args = parser.parse_args()

    config = load_json_config(args.config) if args.config else {}
    motor_key = args.motor or config.get('motor_spec') or DEFAULT_MOTOR_SPEC_KEY
    motor_spec = get_motor_spec(motor_key)
    prop_k = calculate_propeller_constant(motor_spec)
    rpm_scale = float(args.rpm_scale if args.rpm_scale is not None else config.get('rpm_scale', 1.0))
    if rpm_scale <= 0:
        print("Error: --rpm-scale must be greater than zero.")
        sys.exit(1)
    MIN_CURRENT_THRESHOLD = float(
        args.min_current if args.min_current is not None
        else config.get('min_current_threshold', MIN_CURRENT_THRESHOLD)
    )
    if MIN_CURRENT_THRESHOLD < 0:
        print("Error: --min-current cannot be negative.")
        sys.exit(1)

    esc_channel_map = normalize_esc_channel_map(config.get('esc_channel_map'))
    throttle_pwm_min = int(config.get('throttle_pwm_min', 1000))
    throttle_pwm_max = int(config.get('throttle_pwm_max', 2000))
    current_scale_rules = normalize_current_scale_rules(
        config.get('current_scale_rules', []),
        pwm_min=throttle_pwm_min,
        pwm_max=throttle_pwm_max
    )
    esc_count_mode = str(config.get('esc_count_mode', 'fixed')).lower()
    esc_count_fixed = config.get('esc_count_fixed')
    active_esc_throttle_pwm = int(config.get('active_esc_throttle_pwm', MIN_THROTTLE_THRESHOLD))
    if esc_count_mode not in ['fixed', 'throttle']:
        print(f"Warning: Unknown esc_count_mode '{esc_count_mode}', defaulting to fixed.")
        esc_count_mode = 'fixed'
    
    # Helper to parse poles string to dict or int
    def parse_poles(p_list):
        # Join list into single string if multiple parts
        if isinstance(p_list, list):
            p_str = " ".join(p_list)
        else:
            p_str = str(p_list)
            
        p_str = p_str.strip().strip('"').strip("'")
        
        # Replace spaces with commas if needed, or just handle commas
        # If user typed "19 21", p_str is "19 21". Split by space.
        # If user typed "19, 21", p_str is "19, 21". Split by comma.
        
        parts = []
        if ',' in p_str:
            parts = [x.strip() for x in p_str.split(',') if x.strip()]
        else:
            parts = p_str.split()
            
        try:
            # Try to convert all parts to int
            int_parts = [int(x) for x in parts]
        except:
             print(f"Error parsing poles: {p_str}")
             sys.exit(1)
             
        if len(int_parts) == 1:
            return int_parts[0]
        else:
            # Return dict: {0: p0, 1: p1, ...}
            return {i: p for i, p in enumerate(int_parts)}

    if not args.filepath:
        # Fallback for drag-and-drop or simple run
        if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
            filepath = sys.argv[1].strip().strip('"').strip("'")
            poles = 14
        else:
            print("Usage: python plot_esc_data.py <path_to_bin_file> [--poles 14] [--rpm-scale 1.0] [--min-current 15]")
            sys.exit(1)
    else:
        filepath = args.filepath.strip().strip('"').strip("'")
        poles = parse_poles(args.poles)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
        
    print(f"\n{'='*50}")
    print(f"ESC Analysis Tool")
    print(f"File: {os.path.basename(filepath)}")
    print(f"Motor Pole Pairs: {poles} (metadata only)")
    print(f"RPM Scale: {rpm_scale:g} (explicit logged-RPM multiplier)")
    print(f"Per-ESC Current Threshold: {MIN_CURRENT_THRESHOLD:g} A")
    print(f"Motor Spec: {motor_spec['name']}")
    if motor_spec.get('note'):
        print(f"Spec Note: {motor_spec['note']}")
    if current_scale_rules:
        print(f"Current Scaling: {describe_current_scale_rules(current_scale_rules)}")
    if esc_count_mode == 'throttle':
        print(f"Active ESC Count: throttle >= {active_esc_throttle_pwm} PWM")
        if not esc_channel_map:
            print("Warning: No esc_channel_map set; active ESC counts may be inaccurate.")
    elif esc_count_fixed is not None:
        print(f"Active ESC Count: fixed {int(esc_count_fixed)}")
    print(f"{'='*50}")
    
    # Try to load from cache first
    all_esc_data = None
    runs = None
    
    if is_cache_valid(filepath, poles, rpm_scale, esc_channel_map):
        all_esc_data, runs = load_from_cache(filepath, poles, rpm_scale)
    
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
        all_esc_data = load_esc_data(filepath, 0, 0, poles, rpm_scale, esc_channel_map)
        
        # Save to cache for next time
        save_to_cache(filepath, all_esc_data, runs, poles, rpm_scale, esc_channel_map)
    
    # Compute run stats (using cached data - FAST)
    stats = []
    if runs and runs[0] != (0, 0):
        print("Computing run stats...")
        stats = analyze_runs_from_cache(runs, all_esc_data, prop_k=prop_k, current_scale_rules=current_scale_rules)
        print_run_table(stats)
    else:
        print("No runs detected. Will analyze full log.")
    
    # State
    current_run_idx = None
    active_escs = []  # Will be auto-detected from data
    detected_esc_count = 0  # Track how many ESCs are in the log
    esc_count_override = None
    esc_count_series = None
    esc_data = {}  # Current run's data (filtered from all_esc_data)
    derived = {}
    filter_rc = DEFAULT_FILTER_RC  # Low-pass filter RC constant (adjustable)
    efficiency_mode = 'pct'        # 'pct' = Motor Efficiency %, 'rpm_w' = RPM/Watt
    
    def filter_data_for_run(start_us, end_us):
        """Filter all_esc_data to just the specified time range."""
        filtered = defaultdict(lambda: {'time_us': [], 'time': [], 'rpm': [], 'volt': [], 'curr': [], 'temp': [], 'throttle': []})
        global_first_time = min(
            (data['time_us'][0] for data in all_esc_data.values() if data.get('time_us')),
            default=0
        )

        for inst, data in all_esc_data.items():
            for i, t_us in enumerate(data['time_us']):
                if start_us and t_us < start_us:
                    continue
                if end_us and t_us > end_us:
                    break
                
                # Recalculate relative time from run start
                # All ESCs share one origin. Late-starting ESCs retain their
                # true offset instead of being shifted back to t=0.
                run_start = start_us if start_us else global_first_time
                t_sec = (t_us - run_start) / 1e6
                
                filtered[inst]['time_us'].append(t_us)
                filtered[inst]['time'].append(t_sec)
                filtered[inst]['rpm'].append(data['rpm'][i])
                filtered[inst]['volt'].append(data['volt'][i])
                filtered[inst]['curr'].append(data['curr'][i])
                filtered[inst]['temp'].append(data['temp'][i])
                filtered[inst]['throttle'].append(data['throttle'][i] if 'throttle' in data and i < len(data['throttle']) else 0)
        
        return filtered
    
    def load_run(idx):
        nonlocal esc_data, derived, current_run_idx, active_escs, detected_esc_count, filter_rc
        nonlocal esc_count_override, esc_count_series
        
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
        derived.update(compute_derived_metrics(esc_data, filter_rc, prop_k=prop_k, current_scale_rules=current_scale_rules))
        
        # Auto-detect ESCs from loaded data
        detected_escs = sorted(esc_data.keys())
        detected_esc_count = len(detected_escs)
        active_escs = detected_escs.copy()

        # Effective ESC count (fixed or throttle-based)
        esc_count_override = detected_esc_count
        esc_count_series = None
        if esc_count_mode == 'fixed':
            if esc_count_fixed is not None:
                esc_count_override = int(esc_count_fixed)
            else:
                # With no explicit fixed override, follow the number of ESCs
                # actually reporting. This handles logs that begin with two
                # ESCs and later transition to four.
                esc_count_series = derived.get('total', {}).get('esc_count') or None
        elif esc_count_mode == 'throttle':
            esc_count_series = build_active_esc_count_series(esc_data, derived['total']['time'], active_esc_throttle_pwm)
            if not esc_count_series or max(esc_count_series) == 0:
                esc_count_series = None
                print("Warning: Active ESC count requested but throttle data is missing. Using detected ESC count.")
        
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
        eff_unit_label = "%" if efficiency_mode == 'pct' else "RPM/W"
        
        print(f"\n{'='*55}")
        print(f"Current: {run_label} | ESC: [{esc_label}] | Filter: {filter_label} | Eff: {eff_unit_label}")
        print(f"Motor: {motor_spec['name']}")
        if isinstance(poles, dict):
            print(f"Pole metadata: Mixed {poles} | RPM scale: {rpm_scale:g}")
        else:
            print(f"Pole metadata: {poles} pairs | RPM scale: {rpm_scale:g}")
        print(f"{'='*55}")
        print("[1] Plot ESC Basics (RPM, Volt, Curr, Temp)")
        print("[2] Plot Power (Total Current & Power)")
        print(f"[3] Plot Efficiency ({eff_unit_label})")
        print("[4] Voltage Sag Analysis")
        print(f"[5] Benchmark vs {motor_spec['name']}")
        print("[6] System Analysis (Voltage-Efficiency)")
        print("[7] Change Run")
        print("[8] Filter ESCs")
        print("[9] Export to CSV")
        print(f"[0] Adjust Low-Pass Filter (RC: {filter_label})")
        print(f"[e] Toggle Efficiency Unit")
        print(f"[m] Select Motor/Prop Comparison")
        print(f"[p] Set Pole-Pair Metadata (no RPM scaling)")
        print(f"[s] Set Explicit RPM Scale (current: {rpm_scale:g})")
        print(f"[w] Voltage vs Time / Energy Used (click two times)")
        print(f"[d] 3D Efficiency vs ESC Power / Time (voltage color)")
        print(f"[g] Efficiency response curves by voltage band")
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
            plot_efficiency(esc_data, derived, active_escs, title, f"{base_save}_efficiency.png", efficiency_mode)
        elif choice == '4':
            plot_voltage_sag(
                esc_data,
                derived,
                active_escs,
                title,
                f"{base_save}_vsag.png",
                esc_count_override,
                esc_count_series
            )
        elif choice == '5':
            plot_benchmark(esc_data, derived, active_escs, title, f"{base_save}_benchmark.png", motor_spec, efficiency_mode)
        elif choice == '6':
            plot_system_analysis(
                esc_data,
                derived,
                active_escs,
                title,
                f"{base_save}_sysanalysis.png",
                esc_count_override,
                efficiency_mode,
                esc_count_series
            )
        elif choice == 'd':
            plot_efficiency_power_time_3d(
                esc_data,
                derived,
                active_escs,
                title,
                f"{base_save}_efficiency_power_time_3d.png",
                efficiency_mode
            )
        elif choice == 'g':
            plot_efficiency_power_voltage_curves(
                esc_data,
                derived,
                active_escs,
                title,
                f"{base_save}_efficiency_power_voltage_curves.png",
                efficiency_mode
            )
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
                derived.update(compute_derived_metrics(esc_data, filter_rc, prop_k=prop_k, current_scale_rules=current_scale_rules))
                print("Done! Metrics recomputed with new filter.")
            except:
                print("Invalid input. Keeping current filter.")
        elif choice == 'e':
            # Toggle efficiency unit
            efficiency_mode = 'rpm_w' if efficiency_mode == 'pct' else 'pct'
            print(f"Switched efficiency unit to: {'RPM/Watt' if efficiency_mode == 'rpm_w' else 'Motor Efficiency (%)'}")
        elif choice == 'm':
            print("Available motor/prop comparisons:")
            motor_keys = sorted(MOTOR_SPECS.keys())
            for idx, key in enumerate(motor_keys, start=1):
                spec = MOTOR_SPECS[key]
                default_mark = " (default)" if key == DEFAULT_MOTOR_SPEC_KEY else ""
                print(f"  {idx}. {key}: {spec['name']} / {spec['prop']}{default_mark}")
            val = input("Motor number or key: ").strip()
            try:
                if val.isdigit():
                    selection = int(val)
                    if selection < 1 or selection > len(motor_keys):
                        raise ValueError("motor number out of range")
                    selected_key = motor_keys[selection - 1]
                else:
                    selected_key = val
                if selected_key not in MOTOR_SPECS:
                    raise ValueError("unknown motor comparison")
                motor_spec = get_motor_spec(selected_key)
                motor_key = selected_key
                prop_k = calculate_propeller_constant(motor_spec)
                derived.clear()
                derived.update(compute_derived_metrics(
                    esc_data, filter_rc, prop_k=prop_k,
                    current_scale_rules=current_scale_rules
                ))
                print(f"Comparison updated to {motor_spec['name']} / {motor_spec['prop']}.")
            except (ValueError, IndexError):
                print("Invalid motor selection. Keeping current comparison.")
        elif choice == 'p':
            print(f"Current motor pole-pair metadata: {poles}")
            print("This value documents the motor only; it does not change logged RPM.")
            print("Enter new metadata:")
            print("  - Single int (e.g. '19') for all ESCs")
            print("  - Comma list (e.g. '19,21,19,21') for ESC 0,1,2,3")
            print("  - Key:Value (e.g. '0:19, 1:21') for specific instances")
            val = input("> ").strip().strip('"').strip("'")
            
            try:
                if ':' in val:
                    # Parse key:value pairs
                    new_poles = {}
                    for item in val.split(','):
                        k, v = item.split(':')
                        new_poles[int(k.strip())] = int(v.strip())
                elif ',' in val:
                    # Parse list
                    parts = [int(x.strip()) for x in val.split(',') if x.strip()]
                    new_poles = {i: p for i, p in enumerate(parts)}
                else:
                    # Parse single int
                    new_poles = int(val)
                    if new_poles < 2: raise ValueError
                
                poles = new_poles
                print(f"Pole-pair metadata updated to {poles}; RPM values are unchanged.")
            except Exception as e:
                print(f"Invalid input: {e}")
        elif choice == 's':
            print(f"Current explicit RPM scale: {rpm_scale:g}")
            print("Enter a direct multiplier for logged RPM (1.0 leaves RPM unchanged):")
            try:
                new_rpm_scale = float(input("> ").strip())
                if new_rpm_scale <= 0:
                    raise ValueError("scale must be greater than zero")
                rpm_scale = new_rpm_scale
                if is_cache_valid(filepath, poles, rpm_scale, esc_channel_map):
                    all_esc_data, _ = load_from_cache(filepath, poles, rpm_scale)
                else:
                    print("Parsing bin file with the new explicit RPM scale...")
                    all_esc_data = load_esc_data(
                        filepath, 0, 0, poles, rpm_scale, esc_channel_map
                    )
                    save_to_cache(
                        filepath, all_esc_data, runs, poles, rpm_scale,
                        esc_channel_map
                    )
                load_run(current_run_idx)
                print(f"RPM scale updated to {rpm_scale:g}.")
            except Exception as e:
                print(f"Invalid RPM scale: {e}")
        elif choice == 'w':
            plot_interactive_voltage_energy_curve(
                esc_data,
                derived,
                active_escs,
                title,
                f"{base_save}_voltage_time_energy.png"
            )
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()
