from train import FEATURES, build_pipeline, make_dataset

def test_synthetic_data_has_expected_columns_and_both_classes():
    data = make_dataset(2000)
    assert set(FEATURES).issubset(data.columns)
    assert set(data["is_fraud"].unique()) == {0, 1}

def test_pipeline_fits_and_returns_probabilities():
    data = make_dataset(2000)
    model = build_pipeline().fit(data[FEATURES], data["is_fraud"])
    probabilities = model.predict_proba(data[FEATURES].head(10))[:, 1]
    assert len(probabilities) == 10
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
