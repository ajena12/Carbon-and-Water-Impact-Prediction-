import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def load_training_frame(data_dir: str) -> pd.DataFrame:
    routes_sf = pd.read_csv(os.path.join(data_dir, "routes_sup_to_fac.csv"))
    routes_fr = pd.read_csv(os.path.join(data_dir, "routes_fac_to_reg.csv"))
    product = pd.read_csv(os.path.join(data_dir, "product.csv"))
    weight_kg_per_unit = float(product["weight_kg_per_unit"].iloc[0])

    # Build per-route samples with a reference load: 1 ton for SF, 1 unit for FR
    # Target: emissions_kg for that reference load
    sf = routes_sf.copy()
    sf["reference_load_tons"] = 1.0
    sf["emissions_kg"] = sf["distance_km"] * sf["ef_kg_per_ton_km"] * sf["reference_load_tons"]
    sf["route_type"] = "S2F"

    fr = routes_fr.copy()
    fr["reference_load_units"] = 1.0
    fr["emissions_kg"] = (
        fr["distance_km"] * fr["ef_kg_per_ton_km"] * (weight_kg_per_unit / 1000.0) * fr["reference_load_units"]
    )
    fr["route_type"] = "F2R"

    # Harmonize columns
    sf = sf[["distance_km", "mode", "emissions_kg", "route_type"]]
    fr = fr[["distance_km", "mode", "emissions_kg", "route_type"]]
    df = pd.concat([sf, fr], ignore_index=True)
    return df


def train_model(data_dir: str, model_dir: str) -> dict:
    os.makedirs(model_dir, exist_ok=True)
    df = load_training_frame(data_dir)
    X = df[["distance_km", "mode", "route_type"]]
    y = df["emissions_kg"]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["distance_km"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["mode", "route_type"]),
        ]
    )
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])
    pipe.fit(X, y)

    preds = pipe.predict(X)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    joblib.dump(pipe, os.path.join(model_dir, "emissions_model.joblib"))

    metrics = {"mae": float(mae), "r2": float(r2), "n": int(len(df))}
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        import json

        json.dump(metrics, f, indent=2)

    print(f"Model trained. MAE={mae:.4f}, R2={r2:.4f}. Saved to {model_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_dir", type=str, default="outputs/model")
    args = parser.parse_args()
    train_model(args.data_dir, args.model_dir)


if __name__ == "__main__":
    main()

