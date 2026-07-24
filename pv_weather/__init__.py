"""Shared analysis code for the German photovoltaic weather project."""

import os

# Recent Windows installations may not provide the legacy command Joblib uses
# to count physical CPU cores. Joblib would fall back to logical cores anyway,
# but emits a noisy warning first. Its current implementation skips that probe
# only when the configured limit is lower than the logical CPU count.
_logical_cpu_count = os.cpu_count() or 1
_loky_cpu_limit = max(1, min(8, _logical_cpu_count - 1))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_loky_cpu_limit))

from .data import BASE_COLUMNS, generate_demo_data, load_project_data, validate_hourly_data
from .features import MODEL_FEATURES, TARGET, add_features, estimate_module_temperature
from .modeling import YieldModelBundle, predict_yield, train_yield_model

__all__ = [
    "BASE_COLUMNS",
    "MODEL_FEATURES",
    "TARGET",
    "YieldModelBundle",
    "add_features",
    "estimate_module_temperature",
    "generate_demo_data",
    "load_project_data",
    "predict_yield",
    "train_yield_model",
    "validate_hourly_data",
]
