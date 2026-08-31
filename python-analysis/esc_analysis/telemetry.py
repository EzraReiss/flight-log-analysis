"""ArduPilot log parsing, run detection, and time-series alignment helpers."""

from collections import defaultdict

import numpy as np
from pymavlink import mavutil

def detect_runs(
        filepath, throttle_threshold=1200, cooldown_sec=10.0,
        motor_channels=None, min_run_sec=15.0):
    """Scans log for RCOU messages to identify active runs.

    Activity is taken only from configured motor outputs. This prevents a servo
    or neutral-centered output from holding the run open. Brief motor spin-ups
    shorter than ``min_run_sec`` are ignored.
    """
    motor_channels = sorted(set(motor_channels or [1, 2, 3, 4]))
    try:
        mlog = mavutil.mavlink_connection(filepath)
    except Exception as e:
        print(f"Error opening file: {e}")
        return []

    runs = []
    in_run = False
    current_run_start = 0
    last_active_time = 0
    last_msg_time = 0
    
    while True:
        try:
            msg = mlog.recv_match(type=['RCOU'])
            if msg is None:
                break
            
            last_msg_time = msg.TimeUS
            throttles = [getattr(msg, f'C{i}', 0) for i in motor_channels]
            active_throttles = [t for t in throttles if t > 900]
            max_throttle = max(active_throttles) if active_throttles else 0

            if max_throttle > throttle_threshold:
                if not in_run:
                    in_run = True
                    current_run_start = msg.TimeUS
                last_active_time = msg.TimeUS
            else:
                if (
                        in_run and last_active_time
                        and (msg.TimeUS - last_active_time) > cooldown_sec * 1e6):
                    active_duration = (last_active_time - current_run_start) / 1e6
                    if active_duration >= min_run_sec:
                        runs.append((current_run_start, last_active_time))
                    in_run = False
                    last_active_time = 0
        except:
            continue
            
    if in_run:
        run_end = last_active_time or last_msg_time
        active_duration = (run_end - current_run_start) / 1e6
        if active_duration >= min_run_sec:
            runs.append((current_run_start, run_end))
         
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

