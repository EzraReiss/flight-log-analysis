#!/usr/bin/env python3
"""
LiPo Flight Time Forecaster for ArduPilot .bin logs
=====================================================
Uses VOLTAGE-BASED SoC estimation - does NOT rely on current readings,
which are inaccurate below ~20A per ESC in typical VTOL hover conditions.

Analysis:
  - LiPo cell voltage → State of Charge mapping
  - Voltage depletion rate → flight time projection
  - Voltage sag vs RPM-proxy (no current needed) → internal resistance estimate
  - Pack configuration auto-detection
  - Per-run and full-log forecasting

Poles are fixed at 14 pole pairs (finalized hardware configuration).

Usage:
    python forecast_flight_time.py <path_to_bin_file>
    python forecast_flight_time.py <path_to_bin_file> --capacity 22000 --cells 12 --cutoff 3.5
"""

import sys
import os
import json
import hashlib
import argparse
import csv as csv_module
import bisect
from collections import defaultdict

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)

try:
    from pymavlink import mavutil
except ImportError:
    print("Error: pymavlink not installed. Run: pip install pymavlink")
    sys.exit(1)

# =============================================================================
# Fixed Configuration
# =============================================================================

POLE_PAIRS = 14          # Finalized motor pole pairs
CACHE_VERSION = "ft_v4"  # Increment when cache format changes

# =============================================================================
# LiPo Discharge Model
# =============================================================================
#
# OCV (Open-Circuit Voltage) per cell vs State of Charge
# Based on typical high-discharge (>50C) LiPo characteristics.
# Under load the terminal voltage sags below this curve; we correct for that
# using the internal resistance estimate derived from the V-vs-RPM analysis.
#
LIPO_OCV_CURVE = [
    # (SoC %, V/cell OCV)
    (100, 4.20),
    (95,  4.15),
    (90,  4.10),
    (85,  4.05),
    (80,  4.00),
    (75,  3.97),
    (70,  3.93),
    (65,  3.90),
    (60,  3.87),
    (55,  3.83),
    (50,  3.79),
    (45,  3.75),
    (40,  3.72),
    (35,  3.68),
    (30,  3.63),
    (25,  3.58),
    (20,  3.52),
    (15,  3.45),
    (10,  3.38),
    (5,   3.25),
    (0,   3.00),
]

_SOC_VALS  = np.array([x[0] for x in LIPO_OCV_CURVE])  # SoC,  descending 100→0
_VCELL_VALS = np.array([x[1] for x in LIPO_OCV_CURVE]) # V/cell, descending

def vcell_to_soc(vcell: float) -> float:
    """Map per-cell OCV to State-of-Charge (%).
    Linear interpolation between known curve points.
    Clamped to 0–100.
    """
    return float(np.interp(vcell, _VCELL_VALS[::-1], _SOC_VALS[::-1]))

def soc_to_vcell(soc: float) -> float:
    """Map State-of-Charge (%) to per-cell OCV."""
    return float(np.interp(soc, _SOC_VALS[::-1], _VCELL_VALS[::-1]))


def detect_cell_count(voltages: list[float]) -> int:
    """Infer LiPo cell count from peak observed voltage.
    Works for 6S, 8S, 10S, 12S, 14S packs.
    """
    peak = max(voltages)
    # Each fully-charged cell is ≤4.25V; cells at storage (~3.85V) or mid-discharge
    # 6S  → 22.2–25.2V nominal
    # 8S  → 29.6–33.6V
    # 10S → 37.0–42.0V
    # 12S → 44.4–50.4V
    # 14S → 51.8–58.8V
    for cells, max_full_v in [(6, 26.0), (8, 34.5), (10, 43.5), (12, 51.0), (14, 60.0)]:
        if peak <= max_full_v:
            return cells
    return 12  # Fallback


# =============================================================================
# Cache helpers (reuse ESC data cache from plot_esc_data.py when available)
# =============================================================================

def get_output_dir(filepath: str) -> str:
    base = os.path.dirname(filepath)
    name = os.path.splitext(os.path.basename(filepath))[0]
    d = os.path.join(base, f"{name}_analysis")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_paths(filepath: str):
    d = get_output_dir(filepath)
    return (
        os.path.join(d, "forecast_esc_cache.csv"),
        os.path.join(d, "forecast_cache_meta.json"),
    )


def cache_valid(filepath: str) -> bool:
    csv_p, meta_p = _cache_paths(filepath)
    if not os.path.exists(csv_p) or not os.path.exists(meta_p):
        return False
    try:
        with open(meta_p) as f:
            meta = json.load(f)
        if meta.get("version") != CACHE_VERSION:
            return False
        if meta.get("bin_mtime") != os.path.getmtime(filepath):
            return False
        if meta.get("bin_size") != os.path.getsize(filepath):
            return False
        return True
    except Exception:
        return False


def load_cache(filepath: str):
    csv_p, meta_p = _cache_paths(filepath)
    print("Loading from cache…")
    df = pd.read_csv(csv_p)
    with open(meta_p) as f:
        meta = json.load(f)

    esc_data = defaultdict(lambda: {"time_us": [], "time": [], "rpm": [], "volt": [], "temp": [], "throttle": []})
    bat_data = {"time": [], "volt": [], "curr": []}

    for inst in df["instance"].unique():
        if inst == -1:
            sub = df[df["instance"] == -1].sort_values("time")
            bat_data["time"]  = sub["time"].tolist()
            bat_data["volt"]  = sub["volt"].tolist()
            bat_data["curr"]  = sub["curr"].tolist() if "curr" in sub.columns else []
        else:
            sub = df[df["instance"] == inst].sort_values("time")
            esc_data[inst] = {
                "time_us":  sub["time_us"].tolist(),
                "time":     sub["time"].tolist(),
                "rpm":      sub["rpm"].tolist(),
                "volt":     sub["volt"].tolist(),
                "temp":     sub["temp"].tolist(),
                "throttle": sub["throttle"].tolist() if "throttle" in sub.columns else [0]*len(sub),
            }

    runs = [tuple(r) for r in meta.get("runs", [])]
    return esc_data, bat_data, runs


def save_cache(filepath: str, esc_data, bat_data, runs):
    csv_p, meta_p = _cache_paths(filepath)
    print("Saving cache…")
    rows = []
    for inst, d in esc_data.items():
        for i in range(len(d["time"])):
            rows.append({
                "instance": inst,
                "time_us":  d["time_us"][i],
                "time":     d["time"][i],
                "rpm":      d["rpm"][i],
                "volt":     d["volt"][i],
                "curr":     0,   # Not used (inaccurate)
                "temp":     d["temp"][i],
                "throttle": d["throttle"][i] if i < len(d["throttle"]) else 0,
            })
    # Store BAT data as instance -1
    for i in range(len(bat_data["time"])):
        rows.append({
            "instance": -1,
            "time_us":  0,
            "time":     bat_data["time"][i],
            "rpm":      0,
            "volt":     bat_data["volt"][i],
            "curr":     bat_data["curr"][i] if i < len(bat_data["curr"]) else 0,
            "temp":     0,
            "throttle": 0,
        })
    pd.DataFrame(rows).to_csv(csv_p, index=False)
    with open(meta_p, "w") as f:
        json.dump({
            "version":   CACHE_VERSION,
            "bin_mtime": os.path.getmtime(filepath),
            "bin_size":  os.path.getsize(filepath),
            "runs":      runs,
        }, f, indent=2)
    print(f"Cached to {os.path.basename(csv_p)}")


# =============================================================================
# Data Parsing
# =============================================================================

def _merge_windows(windows: list, cooldown_us: float) -> list:
    """Merge consecutive (start_us, end_us) windows that are closer than cooldown_us."""
    if not windows:
        return []
    merged = [list(windows[0])]
    for s, e in windows[1:]:
        if s - merged[-1][1] <= cooldown_us:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(w) for w in merged]


def _filter_short(windows: list, min_sec: float) -> list:
    """Drop segments shorter than min_sec (ground vibration / isFlying blips)."""
    return [(s, e) for s, e in windows if (e - s) >= min_sec * 1e6]


def _print_segments(method: str, raw: list, final: list):
    """Print raw vs final segment list so user can see what was kept/merged/dropped."""
    print(f"  Airborne detection method: {method}")
    print(f"  Raw isFlying windows before filtering ({len(raw)}):")
    for s, e in raw:
        dur = (e - s) / 1e6
        kept = any(fs <= s and e <= fe for fs, fe in final) or \
               any((e - s) / 1e6 < 0.001 for fs, fe in [(s, e)])
        mark = "  " if any(fs <= s + 1 and e <= fe + 1 for fs, fe in final) else "XX"
        print(f"    {mark} {dur:6.1f}s  ({s/1e6:.1f}s -> {e/1e6:.1f}s)")
    print(f"  After merge + min-duration filter → {len(final)} flight(s):")
    for i, (s, e) in enumerate(final):
        print(f"    Flight {i+1}: {(e-s)/1e6:.1f}s  ({s/1e6:.1f}s -> {e/1e6:.1f}s)")


def detect_airborne_segments(filepath: str, min_alt_m: float = 0.5,
                              cooldown_sec: float = 20.0,
                              min_flight_sec: float = 15.0) -> list:
    """Detect when the aircraft is actually airborne (not just motors spinning on ground).

    Detection priority:
      1. STAT.isFlying  — ArduPilot's own flight detection (most reliable)
         Note: isFlying can toggle briefly during taxi/vibration.  Short blips
         (< min_flight_sec) and gaps (< cooldown_sec) are filtered/merged out.
      2. GPS.RelAlt > min_alt_m  — GPS height above home
      3. CTUN.Alt > min_alt_m   — barometer / EKF altitude above home
      4. Throttle fallback       — RCOU max channel (legacy, least accurate)

    Multiple takeoffs/landings within a single battery are each their own segment.
    Gaps shorter than cooldown_sec are merged into the surrounding segment.
    Segments shorter than min_flight_sec are discarded as ground noise.

    Returns list of (start_us, end_us) absolute microsecond timestamps.
    """
    cooldown_us = cooldown_sec * 1e6

    # ── Method 1: STAT.isFlying ───────────────────────────────────────────
    windows = []
    in_flight = False
    seg_start = 0
    last_t = 0
    try:
        mlog = mavutil.mavlink_connection(filepath)
        while True:
            msg = mlog.recv_match(type=["STAT"])
            if msg is None:
                break
            last_t = msg.TimeUS
            # isFlying is 1 when airborne; check both field names used across FW versions
            flying = getattr(msg, "isFlying", None)
            if flying is None:
                flying = getattr(msg, "inFlight", None)
            if flying is None:
                break  # STAT doesn't have this field in this log
            flying = bool(flying)
            if flying and not in_flight:
                in_flight = True
                seg_start = msg.TimeUS
            elif not flying and in_flight:
                windows.append((seg_start, msg.TimeUS))
                in_flight = False
    except Exception:
        pass
    if in_flight and seg_start:
        windows.append((seg_start, last_t))

    if windows:
        result = _merge_windows(windows, cooldown_us)
        result = _filter_short(result, min_flight_sec)
        _print_segments("STAT.isFlying", windows, result)
        return result

    # ── Method 2: GPS.RelAlt ──────────────────────────────────────────────
    windows = []
    in_flight = False
    seg_start = 0
    last_t = 0
    try:
        mlog = mavutil.mavlink_connection(filepath)
        while True:
            msg = mlog.recv_match(type=["GPS"])
            if msg is None:
                break
            last_t = msg.TimeUS
            rel_alt = getattr(msg, "RelAlt", None)
            if rel_alt is None:
                rel_alt = getattr(msg, "Alt", None)  # some builds use Alt
            if rel_alt is None:
                break
            airborne = rel_alt > min_alt_m
            if airborne and not in_flight:
                in_flight = True
                seg_start = msg.TimeUS
            elif not airborne and in_flight:
                windows.append((seg_start, msg.TimeUS))
                in_flight = False
    except Exception:
        pass
    if in_flight and seg_start:
        windows.append((seg_start, last_t))

    if windows:
        result = _merge_windows(windows, cooldown_us)
        result = _filter_short(result, min_flight_sec)
        _print_segments(f"GPS.RelAlt > {min_alt_m}m", windows, result)
        return result

    # ── Method 3: CTUN (barometer / EKF alt above home) ───────────────────
    windows = []
    in_flight = False
    seg_start = 0
    last_t = 0
    try:
        mlog = mavutil.mavlink_connection(filepath)
        while True:
            msg = mlog.recv_match(type=["CTUN"])
            if msg is None:
                break
            last_t = msg.TimeUS
            alt = getattr(msg, "Alt", None)
            if alt is None:
                break
            airborne = alt > min_alt_m
            if airborne and not in_flight:
                in_flight = True
                seg_start = msg.TimeUS
            elif not airborne and in_flight:
                windows.append((seg_start, msg.TimeUS))
                in_flight = False
    except Exception:
        pass
    if in_flight and seg_start:
        windows.append((seg_start, last_t))

    if windows:
        result = _merge_windows(windows, cooldown_us)
        result = _filter_short(result, min_flight_sec)
        _print_segments(f"CTUN.Alt > {min_alt_m}m", windows, result)
        return result

    # ── Method 4: Throttle fallback (last resort) ─────────────────────────
    print("  WARNING: No altitude/STAT data found. Falling back to throttle-based detection.")
    return _detect_runs_throttle(filepath, cooldown_sec=cooldown_sec)


def _detect_runs_throttle(filepath: str, throttle_threshold: int = 1200,
                           cooldown_sec: float = 8.0) -> list:
    """Fallback: detect runs from RCOU max throttle channel."""
    try:
        mlog = mavutil.mavlink_connection(filepath)
    except Exception as e:
        print(f"Error opening file: {e}")
        return []

    runs, in_run = [], False
    run_start = potential_end = last_t = 0
    while True:
        try:
            msg = mlog.recv_match(type=["RCOU"])
            if msg is None:
                break
            last_t = msg.TimeUS
            ch = [getattr(msg, f"C{i}", 0) for i in range(1, 9)]
            active = [c for c in ch if c > 900]
            max_t = max(active) if active else 0
            if not in_run:
                if max_t > throttle_threshold:
                    in_run, run_start, potential_end = True, msg.TimeUS, 0
            else:
                if max_t < throttle_threshold:
                    if potential_end == 0:
                        potential_end = msg.TimeUS
                    elif (msg.TimeUS - potential_end) > cooldown_sec * 1e6:
                        runs.append((run_start, msg.TimeUS))
                        in_run = potential_end = 0
                else:
                    potential_end = 0
        except Exception:
            continue
    if in_run:
        runs.append((run_start, last_t))
    return runs


def parse_bin_file(filepath: str):
    """Parse ESC + BAT + RCOU messages from a BIN file."""
    print("Parsing BIN file (ESC + BAT messages)…")

    # ── Pass 1: RCOU throttle lookup ──────────────────────────────────────
    throttle_lut: dict[int, float] = {}
    mlog = mavutil.mavlink_connection(filepath)
    while True:
        try:
            msg = mlog.recv_match(type=["RCOU"])
            if msg is None:
                break
            ch = [getattr(msg, f"C{i}", 0) for i in range(1, 9)]
            valid = [c for c in ch if c > 900]
            throttle_lut[msg.TimeUS] = max(valid) if valid else 0
        except Exception:
            continue

    thr_times = sorted(throttle_lut)

    def get_throttle(ts):
        if not thr_times:
            return 0
        idx = bisect.bisect_left(thr_times, ts)
        if idx == 0:
            t = thr_times[0]
        elif idx >= len(thr_times):
            t = thr_times[-1]
        else:
            tb, ta = thr_times[idx - 1], thr_times[idx]
            t = tb if (ts - tb) < (ta - ts) else ta
        return throttle_lut.get(t, 0)

    # ── Pass 2: ESC messages ──────────────────────────────────────────────
    esc_data = defaultdict(lambda: {"time_us": [], "time": [], "rpm": [], "volt": [], "temp": [], "throttle": []})
    mlog = mavutil.mavlink_connection(filepath)
    first_t = None

    while True:
        try:
            msg = mlog.recv_match(type=["ESC"])
            if msg is None:
                break
            if first_t is None:
                first_t = msg.TimeUS
            t = (msg.TimeUS - first_t) / 1e6
            i = msg.Instance

            # RPM correction: ESC reports at pole-pair count 19 factory default.
            # Actual pole pairs = 14. Corrected RPM = reported * (14/14) = no scale if
            # the ESC is already told poles=14. But to match plot_esc_data default
            # correction formula: rpm * (configured_poles / 14).
            # Since poles ARE 14, scale = 1.0 — raw RPM is correct.
            rpm = msg.RPM  # pole pairs = 14, scale = 14/14 = 1.0

            esc_data[i]["time_us"].append(msg.TimeUS)
            esc_data[i]["time"].append(t)
            esc_data[i]["rpm"].append(rpm)
            esc_data[i]["volt"].append(msg.Volt)
            esc_data[i]["temp"].append(msg.Temp)
            esc_data[i]["throttle"].append(get_throttle(msg.TimeUS))
        except Exception:
            continue

    # ── Pass 3: BAT messages (flight controller battery monitor) ──────────
    bat_data = {"time": [], "volt": [], "curr": []}
    mlog = mavutil.mavlink_connection(filepath)
    bat_first_t = None

    while True:
        try:
            msg = mlog.recv_match(type=["BAT"])
            if msg is None:
                break
            if bat_first_t is None:
                bat_first_t = msg.TimeUS if hasattr(msg, "TimeUS") else None
            # BAT fields: Volt, Curr, EnrgTot, Temp
            t_bat = ((msg.TimeUS - bat_first_t) / 1e6) if bat_first_t else 0
            v = getattr(msg, "Volt", None)
            c = getattr(msg, "Curr", 0)
            if v is not None and v > 5:
                bat_data["time"].append(t_bat)
                bat_data["volt"].append(v)
                bat_data["curr"].append(c)
        except Exception:
            continue

    print(f"  ESC instances: {sorted(esc_data.keys())}, "
          f"ESC points: {sum(len(d['time']) for d in esc_data.values())}, "
          f"BAT points: {len(bat_data['time'])}")

    return esc_data, bat_data


# =============================================================================
# Analysis
# =============================================================================

def avg_esc_voltage(esc_data, instances: list[int]) -> tuple[list[float], list[float]]:
    """Return (times, avg_volt) by averaging across all ESC instances.
    Uses the first ESC's time axis as reference and interpolates others onto it.
    """
    if not instances or not esc_data:
        return [], []

    ref = instances[0]
    ref_times = np.array(esc_data[ref]["time"])
    avg_v = np.array(esc_data[ref]["volt"], dtype=float)

    for inst in instances[1:]:
        if inst not in esc_data or not esc_data[inst]["time"]:
            continue
        interp_v = np.interp(ref_times, esc_data[inst]["time"], esc_data[inst]["volt"])
        avg_v += interp_v

    avg_v /= len(instances)
    return ref_times.tolist(), avg_v.tolist()


def avg_esc_rpm(esc_data, instances: list[int]) -> tuple[list[float], list[float]]:
    """Return (times, avg_rpm) averaged across ESC instances."""
    if not instances or not esc_data:
        return [], []

    ref = instances[0]
    ref_times = np.array(esc_data[ref]["time"])
    avg_r = np.array(esc_data[ref]["rpm"], dtype=float)

    for inst in instances[1:]:
        if inst not in esc_data or not esc_data[inst]["rpm"]:
            continue
        interp_r = np.interp(ref_times, esc_data[inst]["time"], esc_data[inst]["rpm"])
        avg_r += interp_r

    avg_r /= len(instances)
    return ref_times.tolist(), avg_r.tolist()


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def estimate_internal_resistance(volts: np.ndarray, rpm_proxy: np.ndarray,
                                  min_rpm: float = 500) -> tuple[float, float, np.ndarray]:
    """Estimate effective internal resistance proxy from voltage sag vs RPM^3.

    Model: V = V_oc0 - Ri_proxy * RPM^3
    where RPM^3 is proportional to motor shaft power (prop law: P ∝ n³).

    Returns:
        (Ri_proxy, V_oc0, residuals)
    Ri_proxy has units V / (RPM^3), multiply by avg RPM^3 to get typical sag.
    """
    mask = rpm_proxy > min_rpm
    if mask.sum() < 10:
        return 0.0, float(np.mean(volts)), np.zeros_like(volts)

    x = rpm_proxy[mask] ** 3
    y = volts[mask]
    # Linear least squares: y = a - b*x  → [a, b]
    coeffs = np.polyfit(x, y, 1)
    Ri_proxy = coeffs[0]   # slope (negative → voltage drops with load)
    V_oc0    = coeffs[1]   # intercept (estimated no-load voltage)
    resid = y - np.polyval(coeffs, x)
    return float(Ri_proxy), float(V_oc0), resid


def forecast(times: np.ndarray, volts: np.ndarray, cells: int, cutoff_v_cell: float,
             smooth_window: int = 30) -> dict:
    """Extrapolate voltage trajectory to cutoff and compute remaining flight time.

    Strategy:
      1. Smooth voltage trace.
      2. Fit a 1st-order (linear) trend to the LAST third of the data.
         (The final portion is most predictive of near-future depletion rate.)
      3. Also fit a 2nd-order polynomial to the entire trace for a longer-horizon estimate.
      4. Use the linear rate for "short-horizon" forecast from last data timestamp.

    Returns dict with analysis results.
    """
    if len(times) < 10:
        return {}

    times = np.array(times)
    volts = np.array(volts)
    cutoff_total = cutoff_v_cell * cells  # e.g. 3.5 * 12 = 42.0V

    # Current stats
    t_start  = times[0]
    t_end    = times[-1]
    t_span   = t_end - t_start       # seconds of data
    v_start  = float(volts[0])
    v_end    = float(volts[-1])
    v_cells_start = v_start / cells
    v_cells_end   = v_end   / cells
    soc_start = vcell_to_soc(v_cells_start)
    soc_end   = vcell_to_soc(v_cells_end)

    # Smoothed voltage series
    v_smooth = rolling_mean(volts, smooth_window)

    # SoC series (using under-load voltage → slight underestimate of true SoC,
    # but the RATE of change is accurate for forecasting)
    soc_series = np.array([vcell_to_soc(v / cells) for v in volts])

    # ── Linear fit on LAST THIRD of data (most recent depletion rate) ────
    n_last = max(len(times) // 3, 5)
    t_last = times[-n_last:]
    v_last = volts[-n_last:]
    lin_coeffs = np.polyfit(t_last, v_last, 1)   # [slope V/s, intercept]
    dv_dt_linear = lin_coeffs[0]                  # V/s (negative = discharging)

    # Time until reaching cutoff from last data point (linear extrapolation)
    if dv_dt_linear < 0:
        t_to_cutoff_linear = (cutoff_total - v_end) / dv_dt_linear   # seconds
        t_remaining_linear = max(0.0, t_to_cutoff_linear)
    else:
        t_remaining_linear = float("inf")  # not discharging in this window

    # ── Polynomial fit on entire data (longer horizon) ────────────────────
    poly_coeffs = np.polyfit(times, volts, 2)
    poly_fn     = np.poly1d(poly_coeffs)
    # Find roots of poly - cutoff: poly(t) = cutoff_total
    roots = np.roots([poly_coeffs[0], poly_coeffs[1], poly_coeffs[2] - cutoff_total])
    t_cutoff_poly = None
    for r in roots:
        if np.isreal(r) and r.real > t_end:
            if t_cutoff_poly is None or r.real < t_cutoff_poly:
                t_cutoff_poly = r.real
    t_remaining_poly = float(t_cutoff_poly - t_end) if t_cutoff_poly else None

    # ── SoC depletion rate ────────────────────────────────────────────────
    soc_rate = (soc_start - soc_end) / t_span * 60  # %SoC per minute

    return {
        "t_start":            t_start,
        "t_end":              t_end,
        "t_span_min":         t_span / 60.0,
        "v_start":            v_start,
        "v_end":              v_end,
        "v_cells_start":      v_cells_start,
        "v_cells_end":        v_cells_end,
        "soc_start":          soc_start,
        "soc_end":            soc_end,
        "soc_rate_pct_min":   soc_rate,
        "dv_dt_linear_mv_s":  dv_dt_linear * 1000,   # mV/s
        "t_remaining_linear": t_remaining_linear,
        "t_remaining_poly":   t_remaining_poly,
        "cutoff_v_total":     cutoff_total,
        "cutoff_v_cell":      cutoff_v_cell,
        "cells":              cells,
        "smooth_v":           v_smooth.tolist(),
        "soc_series":         soc_series.tolist(),
        "lin_coeffs":         lin_coeffs.tolist(),
        "poly_coeffs":        poly_coeffs.tolist(),
        "poly_fn":            poly_fn,
        "times":              times.tolist(),
        "volts":              volts.tolist(),
    }


# =============================================================================
# Plotting
# =============================================================================

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def setup_style():
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass


def plot_forecast(fc: dict, title: str, save_path: str, rpm_times=None, rpm_vals=None):
    """Master forecast figure: voltage timeline, SoC, depletion rate, projection."""
    setup_style()
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(title, fontsize=14, y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.35)

    t    = np.array(fc["times"])
    v    = np.array(fc["volts"])
    vs   = np.array(fc["smooth_v"])
    soc  = np.array(fc["soc_series"])
    cells = fc["cells"]
    cutoff = fc["cutoff_v_total"]
    poly_fn = fc["poly_fn"]

    t_min = t / 60.0   # seconds → minutes

    # ── Panel 1: Voltage over time ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_min, v,  color="steelblue", lw=0.7, alpha=0.45, label="Raw voltage")
    ax1.plot(t_min, vs, color="steelblue", lw=2.0, label="Smoothed voltage")
    ax1.axhline(cutoff, color="red", ls="--", lw=1.5,
                label=f"Cutoff {fc['cutoff_v_cell']:.1f}V/cell ({cutoff:.1f}V)")

    # Polynomial trend line extended into the future
    t_fc_end = fc.get("t_remaining_poly")
    if t_fc_end and np.isfinite(t_fc_end):
        t_future = np.linspace(t[-1], t[-1] + t_fc_end, 200)
        ax1.plot(t_future / 60.0, poly_fn(t_future), "r--", lw=1.5,
                 alpha=0.6, label="Poly forecast")
    else:
        # Linear extrapolation
        lin = np.poly1d(fc["lin_coeffs"])
        t_rem = fc.get("t_remaining_linear", 0)
        if np.isfinite(t_rem) and t_rem > 0:
            t_future = np.linspace(t[-1], t[-1] + min(t_rem, 900), 200)
            ax1.plot(t_future / 60.0, lin(t_future), "r--", lw=1.5,
                     alpha=0.6, label="Linear forecast")

    # Right axis: per-cell voltage
    ax1r = ax1.twinx()
    ax1r.set_ylabel("Per-cell voltage (V)", color="dimgray")
    ax1r.set_ylim(ax1.get_ylim()[0] / cells, ax1.get_ylim()[1] / cells)
    ax1r.tick_params(axis="y", labelcolor="dimgray")

    ax1.set_ylabel("Pack Voltage (V)")
    ax1.set_xlabel("Time (min)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"Voltage Timeline  [{cells}S pack, {cells} cells]")

    # ── Panel 2: State of Charge ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_min, soc, color="darkorange", lw=1.5, label="SoC (under-load V)")
    ax2.axhline(0, color="red", ls=":", lw=1)
    ax2.set_ylim(-5, 105)
    ax2.set_ylabel("State of Charge (%)")
    ax2.set_xlabel("Time (min)")

    # Annotate start / end
    ax2.annotate(f"Start: {fc['soc_start']:.1f}%", xy=(t_min[0], soc[0]),
                 xytext=(t_min[0] + (t_min[-1] - t_min[0]) * 0.05, soc[0] - 7),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)
    ax2.annotate(f"End: {fc['soc_end']:.1f}%", xy=(t_min[-1], soc[-1]),
                 xytext=(t_min[-1] - (t_min[-1] - t_min[0]) * 0.25, soc[-1] + 7),
                 arrowprops=dict(arrowstyle="->", color="black"), fontsize=9)

    ax2.set_title("Estimated State of Charge")
    ax2.legend(fontsize=9)

    # ── Panel 3: SoC depletion rate ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    window = max(20, len(soc) // 30)
    if len(soc) > window * 2:
        # Rolling derivative of SoC (% per minute)
        dsoc = np.gradient(soc, t / 60.0)
        dsoc_smooth = rolling_mean(dsoc, window)
        ax3.plot(t_min, -dsoc_smooth, color="purple", lw=1.5)
        ax3.axhline(fc["soc_rate_pct_min"], color="green", ls="--", lw=1.5,
                    label=f"Avg rate {fc['soc_rate_pct_min']:.2f}%/min")
        ax3.set_ylabel("Depletion Rate (%SoC / min)")
        ax3.set_xlabel("Time (min)")
        ax3.set_title("Instantaneous SoC Depletion Rate")
        ax3.legend(fontsize=9)
        ax3.set_ylim(bottom=0)
    else:
        ax3.text(0.5, 0.5, "Insufficient data for rate plot",
                 ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("SoC Depletion Rate")

    # ── Panel 4: RPM histogram (throttle usage profile) ───────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    if rpm_vals is not None and len(rpm_vals) > 10:
        rpm_arr = np.array(rpm_vals)
        rpm_active = rpm_arr[rpm_arr > 200]
        if len(rpm_active) > 5:
            ax4.hist(rpm_active, bins=40, color="teal", alpha=0.75, edgecolor="none")
            ax4.axvline(np.median(rpm_active), color="red", lw=2,
                        ls="--", label=f"Median {np.median(rpm_active):.0f} RPM")
            ax4.set_xlabel("RPM")
            ax4.set_ylabel("Count")
            ax4.set_title("RPM Distribution (active: >200 RPM)")
            ax4.legend(fontsize=9)
        else:
            ax4.text(0.5, 0.5, "No active RPM data", ha="center", va="center",
                     transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, "No RPM data available", ha="center", va="center",
                 transform=ax4.transAxes)
    ax4.set_title("RPM Distribution")

    # ── Panel 5: Forecast summary text box ───────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    rem_lin = fc.get("t_remaining_linear", float("inf"))
    rem_poly = fc.get("t_remaining_poly")

    rem_lin_str  = f"{rem_lin/60:.1f} min" if np.isfinite(rem_lin) else "N/A (not discharging)"
    rem_poly_str = f"{rem_poly/60:.1f} min" if (rem_poly and np.isfinite(rem_poly)) else "N/A"

    t_total_lin  = (fc["t_span_min"] + rem_lin/60) if np.isfinite(rem_lin) else None
    t_total_poly = (fc["t_span_min"] + rem_poly/60) if (rem_poly and np.isfinite(rem_poly)) else None
    total_lin_str  = f"{t_total_lin:.1f} min" if t_total_lin else "N/A"
    total_poly_str = f"{t_total_poly:.1f} min" if t_total_poly else "N/A"

    summary = (
        f"  FLIGHT TIME FORECAST\n"
        f"  {'─'*34}\n"
        f"  Pack:          {cells}S LiPo\n"
        f"  Cutoff:        {fc['cutoff_v_cell']:.1f} V/cell ({cutoff:.1f} V total)\n"
        f"\n"
        f"  LOG DATA WINDOW\n"
        f"  {'─'*34}\n"
        f"  Duration:      {fc['t_span_min']:.1f} min\n"
        f"  V start:       {fc['v_start']:.2f} V  ({fc['v_cells_start']:.3f} V/cell)\n"
        f"  V end:         {fc['v_end']:.2f} V  ({fc['v_cells_end']:.3f} V/cell)\n"
        f"  SoC start:     {fc['soc_start']:.1f}%\n"
        f"  SoC end:       {fc['soc_end']:.1f}%\n"
        f"  SoC used:      {fc['soc_start'] - fc['soc_end']:.1f}%\n"
        f"  Avg drain:     {fc['soc_rate_pct_min']:.2f}% SoC/min\n"
        f"  dV/dt:         {fc['dv_dt_linear_mv_s']:.1f} mV/s\n"
        f"\n"
        f"  REMAINING FLIGHT (from log end)\n"
        f"  {'─'*34}\n"
        f"  Linear extrap: {rem_lin_str}\n"
        f"  Poly extrap:   {rem_poly_str}\n"
        f"\n"
        f"  TOTAL FLIGHT TIME ESTIMATE\n"
        f"  {'─'*34}\n"
        f"  Linear:        {total_lin_str}\n"
        f"  Polynomial:    {total_poly_str}\n"
        f"\n"
        f"  NOTE: SoC from under-load voltage.\n"
        f"  Actual SoC is ~2-5% higher (sag correction).\n"
        f"  Current data excluded (inaccurate <20A/ESC).\n"
    )

    ax5.text(0.02, 0.97, summary, transform=ax5.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                       edgecolor="goldenrod", alpha=0.9))
    ax5.set_title("Forecast Summary")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_voltage_sag(esc_data, instances, rpm_times, rpm_vals, cells: int,
                     title: str, save_path: str):
    """Voltage sag analysis using RPM^(1/3) as power proxy (no current needed).

    Since P_prop ∝ RPM³, we can write:
        V_terminal = V_oc − R_eff × I  ≈  V_oc − k × RPM³
    Plotting V vs RPM³ gives a scatter with a negative slope proportional to R_eff.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"{title}\nVoltage Sag Analysis (current not used — RPM³ as power proxy)", fontsize=13)

    # Average voltage at each time step from all instances
    v_times, v_avg = avg_esc_voltage(esc_data, instances)
    if not v_times:
        for ax in axes:
            ax.text(0.5, 0.5, "No ESC data", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.show()
        return

    v_arr = np.array(v_avg)
    t_arr = np.array(v_times)

    # Interpolate avg RPM onto same time axis
    rpm_arr = np.interp(t_arr, rpm_times, rpm_vals) if (rpm_times and rpm_vals) else np.zeros_like(t_arr)

    # Only use data where motors are actively spinning
    active_mask = rpm_arr > 300
    v_act  = v_arr[active_mask]
    rpm_act = rpm_arr[active_mask]

    # Power proxy: RPM³ normalised so numbers are manageable
    rpm3 = rpm_act ** 3
    rpm3_norm = rpm3 / 1e9   # scale to ~1 range for display

    # Panel 1: Voltage vs RPM (colored by time)
    ax = axes[0]
    sc = ax.scatter(rpm_act, v_act, c=t_arr[active_mask], cmap="viridis",
                    alpha=0.35, s=8, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Time (s)")

    # Trend line
    if len(rpm_act) > 20:
        coeffs = np.polyfit(rpm_act, v_act, 1)
        rpm_range = np.linspace(rpm_act.min(), rpm_act.max(), 200)
        ax.plot(rpm_range, np.polyval(coeffs, rpm_range), "r-", lw=2,
                label=f"Trend: {coeffs[0]*1000:.2f} mV/RPM")
        ax.legend(fontsize=9)

    ax.set_xlabel("Avg Motor RPM")
    ax.set_ylabel("Pack Voltage (V)")
    ax.set_title("Voltage vs RPM")

    # Panel 2: Voltage vs RPM³ (linear internal-resistance model)
    ax2 = axes[1]
    Ri_proxy, V_oc0, _ = estimate_internal_resistance(v_act, rpm_act)

    sc2 = ax2.scatter(rpm3_norm, v_act, c=t_arr[active_mask], cmap="plasma",
                      alpha=0.35, s=8, edgecolors="none")
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Time (s)")

    if len(rpm3_norm) > 20 and Ri_proxy != 0:
        x_range = np.linspace(rpm3_norm.min(), rpm3_norm.max(), 200)
        fit_v = V_oc0 + Ri_proxy * x_range * 1e9
        ax2.plot(x_range, fit_v, "r-", lw=2,
                 label=f"V = {V_oc0:.2f} − {abs(Ri_proxy)*1e6:.2f}µV/(RPM³/k)\n"
                        f"Est V_OC ≈ {V_oc0:.2f} V  ({V_oc0/cells:.3f} V/cell)")
        ax2.legend(fontsize=9)

    ax2.set_xlabel("RPM³ / 10⁹  (prop power proxy)")
    ax2.set_ylabel("Pack Voltage (V)")
    ax2.set_title("Voltage Sag Model  [V = V_OC − k·RPM³]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


def plot_per_run_summary(run_forecasts: list[dict], title: str, save_path: str):
    """Bar chart comparing SoC consumed and depletion rate across runs."""
    if not run_forecasts:
        return
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{title}\nPer-Run Battery Summary", fontsize=13)

    labels = [f"Run {i+1}" for i in range(len(run_forecasts))]
    soc_used  = [fc["soc_start"] - fc["soc_end"] for fc in run_forecasts]
    rate      = [fc["soc_rate_pct_min"] for fc in run_forecasts]
    durations = [fc["t_span_min"] for fc in run_forecasts]
    rem_lin   = [fc.get("t_remaining_linear", 0) / 60.0
                 if np.isfinite(fc.get("t_remaining_linear", float("inf"))) else 0
                 for fc in run_forecasts]

    x = np.arange(len(labels))
    w = 0.55

    axes[0].bar(x, soc_used, width=w, color="steelblue", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("%SoC consumed")
    axes[0].set_title("SoC Used per Run")
    for i, v in enumerate(soc_used):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

    axes[1].bar(x, rate, width=w, color="darkorange", alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("%SoC / min")
    axes[1].set_title("Average Depletion Rate")
    for i, v in enumerate(rate):
        axes[1].text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    axes[2].bar(x, durations, width=w, color="green", alpha=0.7, label="Logged")
    axes[2].bar(x, rem_lin, width=w, bottom=durations, color="gold",
                alpha=0.8, label="Remaining (linear fcast)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels)
    axes[2].set_ylabel("minutes")
    axes[2].set_title("Flight Duration: Logged + Remaining")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LiPo flight time forecaster (voltage-based, no current required)"
    )
    parser.add_argument("filepath", nargs="?", help="Path to ArduPilot .BIN file")
    parser.add_argument("--capacity", type=float, default=None,
                        help="Battery capacity in mAh (for informational display; optional)")
    parser.add_argument("--cells", type=int, default=12,
                        help="LiPo cell count S (default: 12)")
    parser.add_argument("--cutoff", type=float, default=3.5,
                        help="Per-cell cutoff voltage in V (default: 3.5)")
    parser.add_argument("--merge", type=float, default=20.0,
                        help="Max gap in seconds between isFlying windows to merge "
                             "into one flight (default: 20)")
    parser.add_argument("--min-flight", type=float, default=15.0,
                        help="Minimum flight duration in seconds to keep as a real "
                             "flight (shorter segments are noise, default: 15)")
    args = parser.parse_args()

    # ── Filepath handling ─────────────────────────────────────────────────
    if args.filepath:
        filepath = args.filepath.strip().strip('"').strip("'")
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        filepath = sys.argv[1].strip().strip('"').strip("'")
    else:
        print("Usage: python forecast_flight_time.py <path_to_bin_file>")
        sys.exit(1)

    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    output_dir = get_output_dir(filepath)
    logname    = os.path.basename(filepath)

    print(f"\n{'='*55}")
    print(f"  LiPo Flight Time Forecaster")
    print(f"  File:   {logname}")
    print(f"  Poles:  {POLE_PAIRS} (fixed)")
    print(f"  Cutoff: {args.cutoff:.1f} V/cell")
    print(f"{'='*55}")

    # ── Load data (cache or parse) ────────────────────────────────────────
    if cache_valid(filepath):
        esc_data, bat_data, runs = load_cache(filepath)
    else:
        print("\nDetecting airborne segments…")
        runs = detect_airborne_segments(filepath,
                                         cooldown_sec=args.merge,
                                         min_flight_sec=args.min_flight)
        esc_data, bat_data = parse_bin_file(filepath)
        save_cache(filepath, esc_data, bat_data, runs)

    if not esc_data:
        print("ERROR: No ESC data found in this log file.")
        sys.exit(1)

    instances   = sorted(esc_data.keys())
    esc_count   = len(instances)
    print(f"\nESC instances detected: {instances}  ({esc_count} ESCs)")

    # ── Cell count detection ──────────────────────────────────────────────
    all_volts = []
    for inst in instances:
        all_volts.extend(esc_data[inst]["volt"])
    all_volts = [v for v in all_volts if v > 5]

    cells = args.cells   # defaults to 12; override with --cells
    peak_v = max(all_volts) if all_volts else 0
    print(f"Cell count: {cells}S  (peak logged voltage: {peak_v:.2f} V → "
          f"auto-detect would give {detect_cell_count(all_volts) if all_volts else '?'}S)")

    # Prefer BAT data for voltage (usually more accurate / higher rate)
    use_bat = len(bat_data["time"]) > 50
    if use_bat:
        print(f"Using BAT monitor data for voltage ({len(bat_data['time'])} samples)")
        v_times_full = bat_data["time"]
        v_volts_full = bat_data["volt"]
    else:
        print("Using ESC voltage data (no BAT monitor messages found)")
        v_times_full, v_volts_full = avg_esc_voltage(esc_data, instances)

    rpm_times_full, rpm_vals_full = avg_esc_rpm(esc_data, instances)

    # ── Decide which data segments to analyse ────────────────────────────
    if not runs:
        segments = [("Full log", 0, 0)]
    else:
        segments  = [(f"Flight {i+1}", s, e) for i, (s, e) in enumerate(runs)]
        segments += [("Full log (all)", 0, 0)]

    # Print a quick summary table
    print(f"\n{'─'*60}")
    print(f"{'#':<4}  {'Label':<14}  {'Duration':>9}  {'V_start':>8}  {'V_end':>8}")
    print(f"{'─'*60}")
    for idx, (label, s_us, e_us) in enumerate(segments):
        if s_us == 0 and e_us == 0:
            mask = [True] * len(v_times_full)
            dur_s = v_times_full[-1] - v_times_full[0] if v_times_full else 0
        else:
            # Filter to run time range using absolute time from ESC time_us
            # v_times_full is in seconds (relative), runs are in microseconds.
            # Use first_time_us from esc_data.
            first_us = esc_data[instances[0]]["time_us"][0]
            s_rel = (s_us - first_us) / 1e6
            e_rel = (e_us - first_us) / 1e6
            mask = [(t >= s_rel) and (t <= e_rel) for t in v_times_full]
            dur_s = e_rel - s_rel

        v_filt = [v for v, m in zip(v_volts_full, mask) if m]
        if v_filt:
            print(f"{idx+1:<4}  {label:<14}  {dur_s:>8.1f}s  {v_filt[0]:>7.2f}V  {v_filt[-1]:>7.2f}V")
        else:
            print(f"{idx+1:<4}  {label:<14}  {'---':>9}  {'---':>8}  {'---':>8}")
    print(f"{'─'*60}")

    # ── Interactive loop ──────────────────────────────────────────────────
    run_forecasts_cache: dict[int, dict] = {}

    def run_analysis(seg_idx: int):
        label, s_us, e_us = segments[seg_idx]
        print(f"\n>>> Analysing: {label}")

        if s_us == 0 and e_us == 0:
            vt = v_times_full
            vv = v_volts_full
            rt = rpm_times_full
            rv = rpm_vals_full
        else:
            first_us = esc_data[instances[0]]["time_us"][0]
            s_rel = (s_us - first_us) / 1e6
            e_rel = (e_us - first_us) / 1e6
            vt = [t for t in v_times_full if s_rel <= t <= e_rel]
            vv = [v for t, v in zip(v_times_full, v_volts_full) if s_rel <= t <= e_rel]
            rt = [t for t in rpm_times_full if s_rel <= t <= e_rel]
            rv = [r for t, r in zip(rpm_times_full, rpm_vals_full) if s_rel <= t <= e_rel]

        if len(vv) < 10:
            print("  Not enough data for this segment.")
            return None

        # Remove flat/stale voltage head (pre-arming)
        vv_arr = np.array(vv)
        vt_arr = np.array(vt)
        # Clip to region where voltage is dropping (throttle above idle)
        rv_arr = np.array(rv) if rv else np.zeros_like(vt_arr)
        rt_arr = np.array(rt) if rt else vt_arr
        rpm_on_vt = np.interp(vt_arr, rt_arr, rv_arr) if len(rt_arr) > 0 else np.zeros_like(vt_arr)
        active = rpm_on_vt > 200
        if active.sum() > 20:
            vt_arr  = vt_arr[active]
            vv_arr  = vv_arr[active]

        # Analyse
        fc = forecast(vt_arr, vv_arr, cells, args.cutoff)
        if not fc:
            print("  Forecast failed (insufficient data).")
            return None

        # Print summary to console
        print(f"  Duration analysed:  {fc['t_span_min']:.1f} min")
        print(f"  Voltage:            {fc['v_start']:.2f}V → {fc['v_end']:.2f}V")
        print(f"  V/cell:             {fc['v_cells_start']:.3f}V → {fc['v_cells_end']:.3f}V")
        print(f"  SoC:                {fc['soc_start']:.1f}% → {fc['soc_end']:.1f}%")
        print(f"  SoC consumed:       {fc['soc_start']-fc['soc_end']:.1f}%")
        print(f"  Depletion rate:     {fc['soc_rate_pct_min']:.2f}% SoC/min")
        rem_l = fc.get("t_remaining_linear", float("inf"))
        rem_p = fc.get("t_remaining_poly")
        if np.isfinite(rem_l):
            print(f"  Remaining (linear): {rem_l/60:.1f} min")
        if rem_p and np.isfinite(rem_p):
            print(f"  Remaining (poly):   {rem_p/60:.1f} min")

        # Plot
        safe_label = label.replace(" ", "_").replace("#", "").replace("/", "-")
        base = os.path.join(output_dir, f"forecast_{safe_label}")

        title_str = f"{logname}  —  {label}  [{cells}S, cutoff {args.cutoff}V/cell]"

        plot_forecast(fc, title_str, f"{base}_forecast.png",
                      rpm_times=rt_arr.tolist() if isinstance(rt_arr, np.ndarray) else rt,
                      rpm_vals=rv_arr.tolist() if isinstance(rv_arr, np.ndarray) else rv)

        plot_voltage_sag(esc_data, instances,
                         rt_arr.tolist() if isinstance(rt_arr, np.ndarray) else rt,
                         rv_arr.tolist() if isinstance(rv_arr, np.ndarray) else rv,
                         cells, title_str, f"{base}_vsag.png")

        run_forecasts_cache[seg_idx] = fc
        return fc

    # ── Menu ──────────────────────────────────────────────────────────────
    while True:
        print(f"\n{'='*55}")
        print(f"  {logname}  |  {cells}S  |  cutoff {args.cutoff:.1f}V/cell")
        if args.capacity:
            print(f"  Capacity: {args.capacity:.0f} mAh")
        print(f"{'='*55}")
        for i, (label, _, _) in enumerate(segments):
            tag = "✓" if i in run_forecasts_cache else " "
            print(f"  [{i+1}]{tag} {label}")
        print(f"  [a]  Analyse ALL runs + summary bar chart")
        print(f"  [c]  Change cutoff voltage (current: {args.cutoff:.1f} V/cell)")
        print(f"  [q]  Quit")
        print(f"{'='*55}")

        choice = input("> ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break
        elif choice == "a":
            print("\nAnalysing all segments…")
            for i in range(len(segments)):
                run_analysis(i)
            # Per-run summary (exclude full-log entry)
            run_fcs = [run_forecasts_cache[i] for i in range(len(runs))
                       if i in run_forecasts_cache]
            if run_fcs:
                bar_path = os.path.join(output_dir, "forecast_all_runs_summary.png")
                plot_per_run_summary(run_fcs, logname, bar_path)
        elif choice == "c":
            try:
                new_cutoff = float(input("New cutoff voltage (V/cell): ").strip())
                if 2.5 <= new_cutoff <= 4.0:
                    args.cutoff = new_cutoff
                    run_forecasts_cache.clear()
                    print(f"Cutoff updated to {args.cutoff:.2f} V/cell. Re-run analysis to apply.")
                else:
                    print("Value out of range [2.5 – 4.0]. Keeping current.")
            except ValueError:
                print("Invalid input.")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(segments):
                    run_analysis(idx)
                else:
                    print("Invalid selection.")
            except ValueError:
                print("Invalid input.")


if __name__ == "__main__":
    main()
