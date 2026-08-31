"""Mutable runtime options shared by interactive analysis workflows."""

from dataclasses import dataclass

from .constants import DEFAULT_MIN_CURRENT_THRESHOLD


@dataclass
class RuntimeOptions:
    """Options that can change during one interactive analysis session."""

    min_current_threshold: float = DEFAULT_MIN_CURRENT_THRESHOLD


options = RuntimeOptions()

