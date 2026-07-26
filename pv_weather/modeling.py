"""Temporal training, uncertainty and prediction utilities for PV yield."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import MONOTONIC_CONSTRAINTS, MODEL_FEATURES, TARGET, add_features


MIN_MODEL_GLOBAL_RADIATION_J_CM2 = 10.0
MAX_MODEL_SOLAR_ZENITH_DEG = 90.0


@dataclass
class YieldModelBundle:
    model: HistGradientBoostingRegressor
    feature_names: list[str]
    metrics: dict[str, float]
    residuals: np.ndarray
    split_timestamp: pd.Timestamp
    training_medians: dict[str, float]
    training_bounds: dict[str, tuple[float, float]]
    feature_importance: dict[str, float]


def _metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def select_pv_relevant_hours(featured: pd.DataFrame) -> pd.DataFrame:
    """Select daylight observations with enough radiation for PV modeling."""
    return featured.loc[
        (featured["global_radiation_j_cm2"] > MIN_MODEL_GLOBAL_RADIATION_J_CM2)
        & (featured["solar_zenith_angle_deg"] < MAX_MODEL_SOLAR_ZENITH_DEG)
    ].copy()


def train_yield_model(frame: pd.DataFrame, test_fraction: float = 0.2) -> YieldModelBundle:
    """Train on PV-relevant daylight hours and test on the newest observations."""
    featured = add_features(frame).replace([np.inf, -np.inf], np.nan)
    pv_relevant = select_pv_relevant_hours(featured)
    usable = pv_relevant.dropna(
        subset=[*MODEL_FEATURES, TARGET]
    ).sort_values("timestamp_utc")
    if len(usable) < 500:
        raise ValueError(
            "Mindestens 500 vollständige PV-relevante Tageslichtstunden werden "
            "für Training und Test benötigt."
        )

    split = int(len(usable) * (1 - test_fraction))
    train, test = usable.iloc[:split], usable.iloc[split:]
    x_train, y_train = train[MODEL_FEATURES], train[TARGET]
    x_test, y_test = test[MODEL_FEATURES], test[TARGET]

    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    model = HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=190,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        monotonic_cst=MONOTONIC_CONSTRAINTS,
        random_state=42,
    ).fit(x_train, y_train)

    prediction = np.clip(model.predict(x_test), 0, 1.2)
    model_metrics = _metrics(y_test, prediction)
    baseline_metrics = _metrics(y_test, baseline.predict(x_test))

    importance_sample = test.tail(min(3_000, len(test)))
    importance = permutation_importance(
        model,
        importance_sample[MODEL_FEATURES],
        importance_sample[TARGET],
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
    )
    feature_importance = {
        name: float(max(value, 0))
        for name, value in zip(MODEL_FEATURES, importance.importances_mean, strict=True)
    }
    return YieldModelBundle(
        model=model,
        feature_names=list(MODEL_FEATURES),
        metrics={
            **{f"model_{key}": value for key, value in model_metrics.items()},
            **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
            "train_rows": float(len(train)),
            "test_rows": float(len(test)),
            "model_rows": float(len(usable)),
            "source_rows": float(len(featured)),
            "excluded_low_light_rows": float(len(featured) - len(pv_relevant)),
            "excluded_incomplete_model_rows": float(len(pv_relevant) - len(usable)),
        },
        residuals=y_test.to_numpy() - prediction,
        split_timestamp=pd.Timestamp(test["timestamp_utc"].iloc[0]),
        training_medians={column: float(x_train[column].median()) for column in MODEL_FEATURES},
        training_bounds={
            column: (float(x_train[column].min()), float(x_train[column].max()))
            for column in MODEL_FEATURES
        },
        feature_importance=feature_importance,
    )


def predict_yield(bundle: YieldModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict normalized PV generation and an empirical 80% interval."""
    featured = add_features(frame)
    for column in bundle.feature_names:
        if column not in featured:
            featured[column] = bundle.training_medians[column]
        featured[column] = featured[column].fillna(bundle.training_medians[column])

    point = np.clip(bundle.model.predict(featured[bundle.feature_names]), 0, 1.2)
    low_residual, high_residual = np.quantile(bundle.residuals, [0.1, 0.9])
    return pd.DataFrame(
        {
            "normalized_pv_prediction": point,
            "lower_80": np.clip(point + low_residual, 0, 1.2),
            "upper_80": np.clip(point + high_residual, 0, 1.2),
        },
        index=frame.index,
    )
