#!/usr/bin/env python3
"""Compatibility entry point for the modular ESC analysis package."""

from esc_analysis import *  # noqa: F401,F403 - preserve historical imports
from esc_analysis.cli import main


if __name__ == "__main__":
    main()
