from experiment_tracking import run_experiment


def test_tracking_offline():
    metrics = run_experiment(tracker="none", seed=42)
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1
