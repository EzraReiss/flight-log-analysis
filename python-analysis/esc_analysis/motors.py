"""Motor and propeller reference data plus interpolation-independent models."""

import sys

MOTOR_SPECS = {
    # MAD M6C10 EEE 200KV with FLUXER PRO 20x6.0 MATT prop, 12S.
    # The published MAD comparison table used an AMPX 60A ESC. The installed
    # AMPX 40A ESC should be treated as the current limit; the motor/prop load
    # and RPM comparison remain the applicable reference.
    "mad_m6c10_200kv_12s": {
        'name': 'MAD M6C10 EEE 200KV (12S)',
        'prop': 'FLUXER PRO 20x6.0 MATT',
        'note': 'MAD reference table uses AMPX 60A; installed configuration uses AMPX 40A',
        'static_test_reference': 'MAD 12S / FLUXER PRO 20x6.0 MATT / AMPX 60A, 25C sea level',
        # Published static-test values keyed by the same throttle rows as data.
        # Thrust is gram-force; torque is shaft/propeller torque in N*m.
        'thrust_gf': {
            30: 1091, 35: 1384, 40: 1757, 45: 2228, 50: 2767,
            55: 3283, 60: 3801, 65: 4291, 70: 4755, 75: 5337,
            80: 5873, 85: 6468, 90: 7178, 95: 7830, 100: 8580,
        },
        'torque_nm': {
            30: 0.235, 35: 0.299, 40: 0.385, 45: 0.491, 50: 0.613,
            55: 0.726, 60: 0.840, 65: 0.945, 70: 1.047, 75: 1.183,
            80: 1.300, 85: 1.450, 90: 1.608, 95: 1.755, 100: 1.922,
        },
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

