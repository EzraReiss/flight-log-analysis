"""Versioned CSV cache storage for parsed ESC telemetry."""

import hashlib
import json
import os
from collections import defaultdict

import pandas as pd

from .config import esc_channel_map_to_meta
from .constants import CACHE_VERSION

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

def is_cache_valid(
        filepath, poles, rpm_scale, esc_channel_map,
        run_detection_config=None):
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

        if meta.get('run_detection') != run_detection_config:
            print("Cache run-detection settings mismatch, reparsing...")
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


def save_to_cache(
        filepath, esc_data, runs, poles, rpm_scale, esc_channel_map,
        run_detection_config=None):
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
            'esc_channel_map': esc_channel_map_to_meta(esc_channel_map),
            'run_detection': run_detection_config,
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Cached {len(df)} data points to {os.path.basename(cache_path)}")
        
    except Exception as e:
        print(f"Cache save error: {e}")

