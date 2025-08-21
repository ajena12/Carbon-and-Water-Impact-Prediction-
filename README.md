## Carbon Footprint Optimizer for Beauty Supply Chain

End-to-end, hackathon-ready project for L'Oréal Sustainable Challenge 2025. It simulates a beauty supply chain, estimates carbon emissions, optimizes routing and sourcing to minimize CO₂ while keeping costs low, and provides an interactive Streamlit dashboard.

### Features
- Synthetic data generator for suppliers, factories, regions, routes, and demand
- Carbon estimation utilities using transport-mode-specific emission factors
- Linear Programming optimizer (PuLP) minimizing total CO₂ subject to demand/capacity
- ML regression model to predict emissions for new routes and scenario forecasting
- EDA scripts for insights and KPI tracking
- Streamlit dashboard for upload, optimization, and visualization

### Project Structure
```
data/                       # Generated CSV datasets
outputs/                    # Analysis charts and results
src/
  cfo/
    data/
    utils/
    optimizer/
    ml/
    eda/
    app/
```

### Quickstart
1) Install dependencies
```
pip install -r requirements.txt
```

2) Generate synthetic data
```
PYTHONPATH=src python -m cfo.data.generate_data --out_dir data
```

3) Run EDA and smoke test optimization
```
PYTHONPATH=src python -m cfo.eda.eda --data_dir data --out_dir outputs
PYTHONPATH=src python -m cfo.optimizer.lp_optimizer --data_dir data --results_path outputs/optimization_results.csv
```

4) Train ML model for emissions prediction
```
PYTHONPATH=src python -m cfo.ml.model --data_dir data --model_dir outputs/model
```

5) Launch Streamlit app
```
streamlit run src/cfo/app/streamlit_app.py
```

### Datasets Generated
- suppliers.csv, factories.csv, regions.csv, demand.csv
- product.csv, transport_modes.csv
- routes_sup_to_fac.csv, routes_fac_to_reg.csv

### Objective Function
CO₂ is computed as: Distance_km × Load_tons × EmissionFactor_kgCO2_per_ton_km.

### License
MIT
# Carbon-Footprint-Optimizer-for-Beauty-Supply-Chain