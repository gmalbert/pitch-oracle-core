from precompute_model_diagnostics import _diagnostic_estimator


def test_diagnostics_unwrap_inference_only_calibrator():
    fitted_estimator = object()
    calibrated = type("Calibrated", (), {"estimator": fitted_estimator})()

    assert _diagnostic_estimator(calibrated) is fitted_estimator
    assert _diagnostic_estimator(fitted_estimator) is fitted_estimator
