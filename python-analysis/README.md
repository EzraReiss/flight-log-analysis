# ESC flight-log analysis

Run the tool exactly as before:

```powershell
python plot_esc_data.py path\to\flight.BIN --poles 14 --rpm-scale 1
```

`plot_esc_data.py` is intentionally a small compatibility entry point. The
implementation is organized under `esc_analysis` so new aircraft, telemetry
sources, calculations, and plots can be added without expanding one monolithic
script.

## Package layout

```text
esc_analysis/
  cli.py             interactive application and session orchestration
  config.py          JSON loading and configuration normalization
  constants.py       stable defaults and plot colors
  runtime.py         options changed during an interactive session
  motors.py          motor/propeller reference data and power models
  cache.py           versioned CSV cache paths, validation, load, and save
  telemetry.py       ArduPilot parsing, run detection, and time alignment
  metrics.py         filtering, synchronized totals, efficiency, Wh, run stats
  plotting/
    common.py        shared plot styling and time-range behavior
    overview.py      basic, power, efficiency, system, benchmark, and CSV views
    sag.py           throttle-step resistance and voltage-sag analysis
    efficiency.py    efficiency response by voltage band
    energy.py        voltage-time selection and energy integration
    hover.py         hover detection, thrust balance, CG, weight, and moments
```

The modules depend inward on configuration, telemetry, and metrics. Plotting
contains presentation only, and the CLI composes the workflows. The cache
version and existing command-line/menu interface remain compatible with the
pre-refactor implementation.

## Regression checks

Run the dependency-free unit suite from the repository root:

```powershell
python -m unittest discover -s python-analysis/tests -v
```

For a full log regression, load `00000025-copy.BIN`, select Run 2, and choose
`h`. The expected baseline is two detected runs, Run 2 energy of about 25.60 Wh,
and a roughly 25.2-second accepted hover window.
