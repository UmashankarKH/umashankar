"""
umashankar.dashboard
====================
A Streamlit UI/UX dashboard for the umashankar library.

Launch it two ways:
    1. From the command line, after `pip install umashankar`:
           umashankar
    2. Directly with Streamlit (useful during development):
           streamlit run src/umashankar/dashboard.py
"""

import sys
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

from umashankar.core import Umashankar


def _run_app():
    st.set_page_config(page_title="Umashankar Dashboard", layout="wide")

    st.title("🔮 Umashankar — Zero-Shot Tabular ML Dashboard")
    st.caption("Upload a CSV, pick a target column, and get instant predictions — powered by TabFM.")

    with st.sidebar:
        st.header("1. Upload data")
        uploaded = st.file_uploader("CSV file", type=["csv"])
        st.header("2. Configure")
        task = st.radio("Task type", ["classification", "regression"])
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)

    if uploaded is None:
        st.info("👈 Upload a CSV file to get started.")
        return

    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox("3. Choose target column", df.columns, index=len(df.columns) - 1)

    if st.button("🚀 Run Umashankar", type="primary"):
        with st.spinner("Fitting model and generating predictions..."):
            X = df.drop(columns=[target])
            y = df[target]
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            model = Umashankar(task=task)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = model.score(X_test, y_test)

        st.success("Done!")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Metrics")
            for name, value in metrics.items():
                st.metric(name, f"{value:.4f}")

        with col2:
            st.subheader("Predictions vs Actual")
            result_df = pd.DataFrame({"actual": y_test.values, "predicted": preds})
            if task == "regression":
                fig = px.scatter(
                    result_df, x="actual", y="predicted",
                    trendline="ols", title="Predicted vs Actual"
                )
            else:
                fig = px.histogram(
                    result_df.melt(var_name="type", value_name="value"),
                    x="value", color="type", barmode="group",
                    title="Actual vs Predicted distribution"
                )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full results table")
        st.dataframe(result_df, use_container_width=True)


def main():
    """
    Entry point registered as the `umashankar` console script.
    Re-invokes this same file under `streamlit run` since Streamlit apps
    need to run inside the Streamlit runtime, not as a plain Python function.
    """
    script_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)])


if __name__ == "__main__":
    _run_app()
