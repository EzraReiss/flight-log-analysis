"""Interactive command-line application for ESC log analysis."""

import os
import sys
from collections import defaultdict

from . import runtime
from .cache import is_cache_valid, load_from_cache, save_to_cache
from .config import (
    describe_current_scale_rules,
    load_json_config,
    normalize_current_scale_rules,
    normalize_esc_channel_map,
)
from .constants import DEFAULT_FILTER_RC, DEFAULT_MIN_THROTTLE_THRESHOLD
from .metrics import (
    analyze_runs_from_cache,
    build_active_esc_count_series,
    calculate_propeller_constant,
    combine_runs_data,
    compute_derived_metrics,
)
from .motors import DEFAULT_MOTOR_SPEC_KEY, MOTOR_SPECS, get_motor_spec
from .plotting import (
    export_csv,
    plot_benchmark,
    plot_efficiency,
    plot_efficiency_power_voltage_curves,
    plot_esc_basics,
    plot_hover_balance,
    plot_interactive_voltage_energy_curve,
    plot_power,
    plot_system_analysis,
    plot_voltage_sag,
)
from .cache import get_output_dir
from .telemetry import detect_runs, load_esc_data

def print_run_table(stats):
    print("\n" + "="*99)
    print(f"{'Run':<5} | {'Duration':<10} | {'Max RPM':<9} | {'Max/ESC A':<9} | {'Max Temp':<9} | {'ESC Wh':<9} | {'ESC Ah':<8} | {'Avg Eff':<8}")
    print("-"*99)
    for s in stats:
        print(f"{s['id']:<5} | {s['duration']:<10.1f} | {s['max_rpm']:<9.0f} | {s['max_curr']:<9.1f} | {s['max_temp']:<9.1f} | {s['energy_wh']:<9.2f} | {s.get('charge_ah', 0):<8.2f} | {s['avg_eff']:<8.1f}")
    print("="*99)


def main():

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
    runtime.options.min_current_threshold = float(
        args.min_current if args.min_current is not None
        else config.get('min_current_threshold', runtime.options.min_current_threshold)
    )
    if runtime.options.min_current_threshold < 0:
        print("Error: --min-current cannot be negative.")
        sys.exit(1)

    esc_channel_map = normalize_esc_channel_map(config.get('esc_channel_map'))
    configured_run_channels = config.get('run_detection_channels')
    if configured_run_channels is not None:
        run_detection_channels = sorted({int(value) for value in configured_run_channels})
    elif esc_channel_map:
        run_detection_channels = sorted(set(esc_channel_map.values()))
    else:
        run_detection_channels = [1, 2, 3, 4]
    run_detection_config = {
        'channels': run_detection_channels,
        'throttle_threshold_pwm': float(config.get('run_throttle_threshold_pwm', 1200.0)),
        'cooldown_sec': float(config.get('run_cooldown_sec', 10.0)),
        'min_run_sec': float(config.get('run_min_duration_sec', 15.0)),
    }
    throttle_pwm_min = int(config.get('throttle_pwm_min', 1000))
    throttle_pwm_max = int(config.get('throttle_pwm_max', 2000))
    current_scale_rules = normalize_current_scale_rules(
        config.get('current_scale_rules', []),
        pwm_min=throttle_pwm_min,
        pwm_max=throttle_pwm_max
    )
    esc_count_mode = str(config.get('esc_count_mode', 'fixed')).lower()
    esc_count_fixed = config.get('esc_count_fixed')
    active_esc_throttle_pwm = int(config.get('active_esc_throttle_pwm', DEFAULT_MIN_THROTTLE_THRESHOLD))
    sag_min_throttle_drop_pwm = float(config.get('sag_min_throttle_drop_pwm', 60.0))
    hover_config = {
        'min_duration_sec': float(config.get('hover_min_duration_sec', 5.0)),
        'min_rpm': float(config.get('hover_min_rpm', 3000.0)),
        'max_vertical_speed_mps': float(config.get('hover_max_vertical_speed_mps', 0.5)),
        'max_horizontal_speed_mps': float(config.get('hover_max_horizontal_speed_mps', 2.0)),
        'min_altitude_m': float(config.get('hover_min_altitude_m', 1.0)),
        'max_tilt_deg': float(config.get('hover_max_tilt_deg', 15.0)),
        'air_temperature_c': float(config.get('hover_air_temperature_c', 25.0)),
        'motor_half_length_m': config.get('hover_motor_half_length_m'),
        'motor_half_width_m': config.get('hover_motor_half_width_m'),
    }

    def get_sag_rpm_limit():
        """Use the known 6000 RPM current-telemetry limit only for M6C10."""
        configured = config.get('sag_max_reliable_rpm', '__auto__')
        if configured == '__auto__':
            return 6000.0 if motor_key == 'mad_m6c10_200kv_12s' else None
        if configured is None or str(configured).strip().lower() in ('none', 'off', 'false'):
            return None
        return float(configured)

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
    print(f"Per-ESC Current Threshold: {runtime.options.min_current_threshold:g} A")
    print(f"Motor Spec: {motor_spec['name']}")
    print(
        f"Run Detection: RCOU {run_detection_channels}, "
        f">{run_detection_config['throttle_threshold_pwm']:g} PWM, "
        f">={run_detection_config['min_run_sec']:g}s"
    )
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
    
    if is_cache_valid(
            filepath, poles, rpm_scale, esc_channel_map,
            run_detection_config):
        all_esc_data, runs = load_from_cache(filepath, poles, rpm_scale)
    
    # If cache miss or invalid, parse the full file
    if all_esc_data is None:
        print("\nParsing bin file (first time, this may take a moment)...")
        
        # Detect runs
        print("Scanning for runs...")
        runs = detect_runs(
            filepath,
            throttle_threshold=run_detection_config['throttle_threshold_pwm'],
            cooldown_sec=run_detection_config['cooldown_sec'],
            motor_channels=run_detection_config['channels'],
            min_run_sec=run_detection_config['min_run_sec'],
        )
        
        if not runs:
            print("No runs detected.")
            runs = [(0, 0)]  # Placeholder for full log
        
        # Load ALL ESC data from full file
        print("Loading all ESC data...")
        all_esc_data = load_esc_data(filepath, 0, 0, poles, rpm_scale, esc_channel_map)
        
        # Save to cache for next time
        save_to_cache(
            filepath, all_esc_data, runs, poles, rpm_scale, esc_channel_map,
            run_detection_config)
    
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
            print(f"\nCombining all runs with current >= {runtime.options.min_current_threshold}A filter...")
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
        print(f"  'c' = All runs combined (filtered >{int(runtime.options.min_current_threshold)}A, no gaps)")
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
            run_label = f"All Runs Combined (>{int(runtime.options.min_current_threshold)}A)"
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
        print(f"[g] Efficiency response curves by voltage band")
        print(f"[h] Hover Balance / CG / Weight Estimate")
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
                esc_count_series,
                sag_min_throttle_drop_pwm,
                get_sag_rpm_limit(),
                current_scale_rules
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
        elif choice == 'g':
            plot_efficiency_power_voltage_curves(
                esc_data,
                derived,
                active_escs,
                title,
                f"{base_save}_efficiency_power_voltage_curves.png",
                efficiency_mode
            )
        elif choice == 'h':
            plot_hover_balance(
                filepath,
                esc_data,
                active_escs,
                title,
                f"{base_save}_hover_balance.png",
                motor_spec,
                hover_config
            )
        elif choice == '7':
            # Change Run
            print_run_table(stats) if runs[0] != (0,0) else print("No runs to select.")
            print(f"  'a' = Full log | 'c' = All runs combined (filtered >{int(runtime.options.min_current_threshold)}A)")
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
                if is_cache_valid(
                        filepath, poles, rpm_scale, esc_channel_map,
                        run_detection_config):
                    all_esc_data, _ = load_from_cache(filepath, poles, rpm_scale)
                else:
                    print("Parsing bin file with the new explicit RPM scale...")
                    all_esc_data = load_esc_data(
                        filepath, 0, 0, poles, rpm_scale, esc_channel_map
                    )
                    save_to_cache(
                        filepath, all_esc_data, runs, poles, rpm_scale,
                        esc_channel_map, run_detection_config
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
