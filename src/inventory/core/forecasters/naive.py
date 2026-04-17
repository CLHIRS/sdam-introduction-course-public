"""Compatibility wrapper for naive forecasters under inventory.core."""

from inventory.forecasters.naive import NaiveForecaster, RollingMeanForecaster

__all__ = ["NaiveForecaster", "RollingMeanForecaster"]