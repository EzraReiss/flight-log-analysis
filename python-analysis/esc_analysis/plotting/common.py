"""Shared Matplotlib styling and plot-range helpers."""

import matplotlib.pyplot as plt

from .. import runtime

def setup_style():
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('ggplot')
        except:
            pass


def get_active_time_range(esc_data, threshold=None):
    """Find time range where current exceeds threshold (crops startup/shutdown).
    
    Returns (start_time, end_time) in seconds, or (None, None) if no valid data.
    This helps avoid Y-axis scaling issues from 0-value startup/shutdown periods.
    """
    if threshold is None:
        threshold = runtime.options.min_current_threshold

    all_times_above = []
    
    for inst, data in esc_data.items():
        for t, c in zip(data['time'], data['curr']):
            if c >= threshold:
                all_times_above.append(t)
    
    if not all_times_above:
        return None, None
    
    return min(all_times_above), max(all_times_above)
