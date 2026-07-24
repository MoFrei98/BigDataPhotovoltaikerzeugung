from pv_weather.data import generate_demo_data
from pv_weather.modeling import predict_yield, train_yield_model


def test_temporal_model_and_prediction_interval():
    data = generate_demo_data("2023-01-01", "2023-06-30")
    bundle = train_yield_model(data)
    scenario = data.tail(2).drop(
        columns=["installed_pv_capacity_mw", "pv_generation_mwh"]
    )
    result = predict_yield(bundle, scenario)

    assert bundle.metrics["test_rows"] > 0
    assert bundle.metrics["model_mae"] < bundle.metrics["baseline_mae"]
    assert result["normalized_pv_prediction"].between(0, 1.2).all()
    assert (result["lower_80"] <= result["upper_80"]).all()
    assert set(bundle.feature_importance) == set(bundle.feature_names)
