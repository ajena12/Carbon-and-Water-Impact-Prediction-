import os
import io
import json
import pandas as pd
import streamlit as st

from cfo.optimizer.lp_optimizer import optimize_min_co2
from cfo.ml.model import train_model


st.set_page_config(page_title="Carbon Footprint Optimizer", layout="wide")

st.title("Carbon Footprint Optimizer for Beauty Supply Chain")
st.caption("Simulate, analyze, and optimize sourcing and shipping to reduce CO₂ emissions.")


@st.cache_data
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


with st.sidebar:
    st.header("Data Inputs")
    data_dir = st.text_input("Data directory", value="data")
    cost_budget_multiplier = st.slider("Cost budget multiplier", 0.5, 1.5, 1.0, 0.05)
    run_optimizer = st.button("Optimize")


def load_all(data_dir: str):
    files = [
        "suppliers.csv",
        "factories.csv",
        "regions.csv",
        "demand.csv",
        "routes_sup_to_fac.csv",
        "routes_fac_to_reg.csv",
        "product.csv",
    ]
    dfs = {}
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            dfs[f] = pd.read_csv(path)
    return dfs


tabs = st.tabs(["Overview", "Data", "Optimization", "ML & Forecasting"])

with tabs[0]:
    st.subheader("Project Overview")
    st.markdown(
        "- Estimate CO₂ using transport-mode-specific factors.\n"
        "- Optimize S→F (material) and F→R (product) flows to minimize total CO₂ under cost and capacity.\n"
        "- Train ML model to predict emissions for new routes and run what-if scenarios."
    )

with tabs[1]:
    st.subheader("Current Data Snapshot")
    dfs = load_all(data_dir)
    if not dfs:
        st.warning("No CSVs found in the data directory. Generate data first.")
    else:
        for name, df in dfs.items():
            st.markdown(f"**{name}**")
            st.dataframe(df.head(50))

with tabs[2]:
    st.subheader("Optimization")
    if run_optimizer:
        results_path = os.path.join("outputs", "optimization_results.csv")
        df = optimize_min_co2(data_dir, results_path, cost_budget_multiplier)
        st.success("Optimization complete.")
        st.write(f"Total CO₂ (kg): {df.attrs.get('total_co2_kg', float('nan')):.2f}")
        st.write(f"Total Cost (USD): {df.attrs.get('total_cost_usd', float('nan')):,.2f}")
        st.dataframe(pd.read_csv(results_path))
    else:
        st.info("Click Optimize in the sidebar to run the solver.")

with tabs[3]:
    st.subheader("Train Emissions Model")
    if st.button("Train Model"):
        metrics = train_model(data_dir, os.path.join("outputs", "model"))
        st.json(metrics)
    st.markdown("Use the trained model to estimate emissions for hypothetical routes.")
