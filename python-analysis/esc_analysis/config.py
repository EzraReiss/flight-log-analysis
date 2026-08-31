"""Configuration loading and normalization helpers."""

import json
import os
import sys

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

