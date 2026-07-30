import pandas as pd
from sklearn.datasets import load_iris

from umashankar import quick_run


def test_quick_run_classification():
    data = load_iris(as_frame=True)
    df = data.frame  # includes 'target' column
    model, metrics, X_test, y_test, preds = quick_run(df, target="target", task="classification")
    assert "accuracy" in metrics
    assert len(preds) == len(y_test)
