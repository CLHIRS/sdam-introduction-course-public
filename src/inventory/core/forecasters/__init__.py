"""Compatibility re-exports for forecasters under inventory.core."""

from inventory.forecasters.base import DemandForecaster
from inventory.forecasters.naive import NaiveForecaster, RollingMeanForecaster

__all__ = ["DemandForecaster", "NaiveForecaster", "RollingMeanForecaster"]