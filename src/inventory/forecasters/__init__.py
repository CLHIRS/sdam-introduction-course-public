"""Forecasting interfaces/implementations used by inventory DLA/CFA examples."""

from inventory.forecasters.base import DemandForecaster
from inventory.forecasters.factory import (
	FitConfig,
	fit_ml_forecaster_from_exogenous,
	forecast_with_default_state,
	make_adapter,
	make_ml_forecaster,
)
from inventory.forecasters.ml import (
	MlAr1RegimeDemandForecaster,
	MlDemandForecaster,
	MlRegimeDemandForecaster,
	MultiRegimeAr1FeatureAdapter,
	MultiRegimeFeatureAdapter,
	QuantileBoostingRegimeDemandForecaster,
	RegimeFeatureAdapter,
	SeasonalFeatureAdapter,
)
from inventory.forecasters.naive import (
	ConstantMeanForecaster,
	ExogenousAwareMeanForecaster,
	NaiveForecaster,
	RollingMeanForecaster,
	ExpertDemandForecasterConstant350,
)
from inventory.forecasters.path import (
	ConstantMeanPathForecaster,
	DemandPathForecaster,
	ExogenousMeanPathForecaster,
	SeasonalSinMeanPathForecaster,
)
from inventory.forecasters.ts import EtsDemandForecaster, SarimaxRegimeDemandForecaster

__all__ = [
	"DemandForecaster",
	"ConstantMeanForecaster",
	"NaiveForecaster",
	"RollingMeanForecaster",
	"ExogenousAwareMeanForecaster",
	"ExpertDemandForecasterConstant350",
	"SeasonalFeatureAdapter",
	"RegimeFeatureAdapter",
	"MultiRegimeFeatureAdapter",
	"MultiRegimeAr1FeatureAdapter",
	"MlDemandForecaster",
	"MlRegimeDemandForecaster",
	"QuantileBoostingRegimeDemandForecaster",
	"SarimaxRegimeDemandForecaster",
	"MlAr1RegimeDemandForecaster",
	"DemandPathForecaster",
	"ConstantMeanPathForecaster",
	"SeasonalSinMeanPathForecaster",
	"ExogenousMeanPathForecaster",
	"EtsDemandForecaster",
	"FitConfig",
	"make_adapter",
	"make_ml_forecaster",
	"fit_ml_forecaster_from_exogenous",
	"forecast_with_default_state",
]
