"""Overview, power, efficiency, system, benchmark, and export workflows."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from .. import runtime
from ..constants import COLORS
from ..metrics import build_synchronized_total_series
from .common import get_active_time_range, setup_style

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
    fig.suptitle(f'{title_prefix} - All Runs Combined (Current >= {runtime.options.min_current_threshold}A)', fontsize=14)
    
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
    fig.suptitle(f'{title_prefix} - {ylabel} [Current >= {runtime.options.min_current_threshold}A]', fontsize=14)
    
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

def plot_system_analysis(esc_data, derived, active_escs, title_prefix, save_path, esc_count, mode='pct', esc_count_series=None):
    """Comprehensive system analysis showing voltage, current, power, and efficiency relationships.
    
    Creates 4 subplots:
    1. Voltage vs Efficiency (based on mode) - colored by current
    2. Power vs Efficiency (based on mode) - shows efficiency drop at high power
    3. ESC Input Power vs Efficiency - per-ESC scatter
    4. Voltage vs Total Power - shows power delivery at different voltage levels
    
    Args:
        mode: 'pct' for Motor Efficiency (%), 'rpm_w' for RPM/Watt
    """
    min_total_current = runtime.options.min_current_threshold * esc_count
    count_label = f"{esc_count}"
    if esc_count_series:
        valid_counts = [c for c in esc_count_series if c > 0]
        if valid_counts:
            min_count = min(valid_counts)
            max_count = max(valid_counts)
            count_label = f"{min_count}-{max_count}" if min_count != max_count else f"{min_count}"
            min_total_current = runtime.options.min_current_threshold * min_count
    
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
            if curr < runtime.options.min_current_threshold:
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
    # Plot raw samples only; the voltage-banded response view provides the
    # separate summarized curves when those are needed.
    ax = axs[1, 0]
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
            edgecolors='none', label=f'ESC {i}'
        )
    
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
        point_threshold = runtime.options.min_current_threshold * max(active_count, 1) * 40
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
    fig.suptitle(f'{title_prefix} - Motor Benchmark [Current >= {runtime.options.min_current_threshold}A]', fontsize=14)
    
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
            valid_power = [p for p, c in zip(power, curr) if c >= runtime.options.min_current_threshold]
            valid_rpm = [r for r, c in zip(rpm, curr) if c >= runtime.options.min_current_threshold]
            
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
                if c >= runtime.options.min_current_threshold and e is not None and p > 10:
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

