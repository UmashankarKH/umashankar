# umashankar

Zero-shot tabular machine learning, made easy — with a built-in UI/UX dashboard.

Built on top of [Google's TabFM](https://github.com/google-research/tabfm), a pretrained
tabular foundation model that predicts on new data via in-context learning, with no
training, tuning, or feature engineering required.

## Install

TabFM (Google's pretrained tabular foundation model) is not published on PyPI,
so it must be installed manually first, choosing a JAX or PyTorch backend:

```bash
git clone https://github.com/google-research/tabfm.git
cd tabfm
pip install -e .[pytorch]   # or .[jax]
cd ..
```

Then install umashankar (requires Python >= 3.11):

```bash
pip install umashankar
```

## Use it as a library

```python
from umashankar import Umashankar

model = Umashankar(task="classification")   # or "regression"
model.fit(X_train, y_train)
preds = model.predict(X_test)
print(model.score(X_test, y_test))
```

Or the one-liner version on a dataframe:

```python
from umashankar import quick_run
import pandas as pd

df = pd.read_csv("my_data.csv")
model, metrics, X_test, y_test, preds = quick_run(df, target="label", task="classification")
print(metrics)
```

## Use the dashboard

After installing, launch the UI from your terminal:

```bash
umashankar
```

This opens a Streamlit dashboard in your browser where you can upload a CSV,
pick classification or regression, choose your target column, and get instant
predictions with metrics and charts.

## Notes

- TabFM's pretrained weights are released under a **non-commercial license** —
  check that this fits your use case before deploying commercially.
- Classification currently supports up to 10 classes.
- Memory usage scales with training set size, since all rows are passed in as
  context (relevant for CPU-only environments).

## License

Apache-2.0 for this package's code. TabFM weights carry their own license (see above).
