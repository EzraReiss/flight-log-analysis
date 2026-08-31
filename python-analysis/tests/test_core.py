"""Fast regression tests for the modular analysis core."""

import pathlib
import sys
import unittest


ANALYSIS_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from esc_analysis.config import get_current_scale, normalize_current_scale_rules
from esc_analysis.metrics import build_synchronized_total_series
from esc_analysis.motors import (
    DEFAULT_MOTOR_SPEC_KEY,
    calculate_propeller_constant,
    get_motor_spec,
)
from esc_analysis.plotting.energy import calculate_energy_between_times


def esc_series(current):
    return {
        "time": [0.0, 1.0, 2.0],
        "volt": [10.0, 10.0, 10.0],
        "curr": [current, current, current],
        "temp": [25.0, 25.0, 25.0],
    }


class ConfigurationTests(unittest.TestCase):
    def test_percentage_current_scale_rule(self):
        rules = normalize_current_scale_rules([
            {"min_throttle_pct": 50, "scale": 1.25},
        ])
        self.assertEqual(rules[0]["min_pwm"], 1500.0)
        self.assertEqual(get_current_scale(1499, rules), 1.0)
        self.assertEqual(get_current_scale(1500, rules), 1.25)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.esc_data = {0: esc_series(1.0), 1: esc_series(2.0)}
        self.per_esc = {
            0: {"volt_filtered": [10.0] * 3, "curr_filtered": [1.0] * 3},
            1: {"volt_filtered": [10.0] * 3, "curr_filtered": [2.0] * 3},
        }

    def test_synchronized_totals(self):
        total = build_synchronized_total_series(self.esc_data, self.per_esc)
        self.assertEqual(total["curr"], [3.0, 3.0, 3.0])
        self.assertEqual(total["power"], [30.0, 30.0, 30.0])
        self.assertEqual(total["esc_count"], [2, 2, 2])

    def test_energy_integration(self):
        result = calculate_energy_between_times(
            self.esc_data, self.per_esc, [0, 1], 0.0, 2.0
        )
        self.assertAlmostEqual(result["energy_wh"], 60.0 / 3600.0)
        self.assertAlmostEqual(result["sum_esc_energy_wh"], 60.0 / 3600.0)
        self.assertAlmostEqual(result["charge_ah"], 6.0 / 3600.0)


class MotorModelTests(unittest.TestCase):
    def test_default_motor_has_valid_propeller_model(self):
        motor = get_motor_spec(DEFAULT_MOTOR_SPEC_KEY)
        self.assertEqual(motor["data"][60][4], 4980)
        self.assertEqual(motor["thrust_gf"][60], 3801)
        self.assertGreater(calculate_propeller_constant(motor), 0.0)


if __name__ == "__main__":
    unittest.main()
