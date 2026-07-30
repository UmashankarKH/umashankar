"""
umashankar
==========
Zero-shot tabular ML made easy, with a built-in UI/UX dashboard.

    from umashankar import Umashankar, quick_run

    model = Umashankar(task="classification")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from .core import Umashankar, quick_run

__version__ = "0.1.0"
__all__ = ["Umashankar", "quick_run", "__version__"]
