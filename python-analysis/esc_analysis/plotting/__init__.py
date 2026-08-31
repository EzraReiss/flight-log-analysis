"""Plotting workflows exposed to the interactive CLI."""

from .efficiency import plot_efficiency_power_voltage_curves
from .energy import calculate_energy_between_times, plot_interactive_voltage_energy_curve
from .hover import analyze_hover_balance, plot_hover_balance
from .overview import (
    export_csv,
    plot_all_runs_combined,
    plot_benchmark,
    plot_efficiency,
    plot_esc_basics,
    plot_power,
    plot_system_analysis,
)
from .sag import plot_voltage_sag

__all__ = [name for name in globals() if name.startswith("plot_") or name in {
    "analyze_hover_balance", "calculate_energy_between_times", "export_csv"
}]

