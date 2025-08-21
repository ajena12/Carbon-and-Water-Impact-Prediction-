import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_kpis(data_dir: str) -> dict:
    routes_sf = pd.read_csv(os.path.join(data_dir, "routes_sup_to_fac.csv"))
    routes_fr = pd.read_csv(os.path.join(data_dir, "routes_fac_to_reg.csv"))
    suppliers = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    factories = pd.read_csv(os.path.join(data_dir, "factories.csv"))
    regions = pd.read_csv(os.path.join(data_dir, "regions.csv"))
    demand = pd.read_csv(os.path.join(data_dir, "demand.csv"))
    product = pd.read_csv(os.path.join(data_dir, "product.csv"))

    weight_kg_per_unit = float(product["weight_kg_per_unit"].iloc[0])

    # Reference flows for KPI estimation (not optimized): deliver demand via nearest factory and nearest supplier
    demand_map = dict(zip(demand.region_id, demand.units_demanded))

    fr_sorted = routes_fr.sort_values(["region_id", "distance_km"]).groupby("region_id").head(1)
    total_units = sum(demand_map.values())
    total_cost = 0.0
    total_co2 = 0.0
    for _, r in fr_sorted.iterrows():
        units = demand_map[r.region_id]
        co2 = r.distance_km * r.ef_kg_per_ton_km * (weight_kg_per_unit / 1000.0) * units
        cost = r.distance_km * r.cost_per_ton_km * (weight_kg_per_unit / 1000.0) * units
        total_co2 += co2
        total_cost += cost

    # Material from suppliers to factories by nearest supplier per factory proportional to factory share
    sf_sorted = routes_sf.sort_values(["factory_id", "distance_km"]).groupby("factory_id").head(1)
    material_kg_per_unit = float(product["material_kg_per_unit"].iloc[0])
    per_factory_units = total_units / len(factories)
    per_factory_tons = (per_factory_units * material_kg_per_unit) / 1000.0
    for _, r in sf_sorted.iterrows():
        co2 = r.distance_km * r.ef_kg_per_ton_km * per_factory_tons
        cost = r.distance_km * r.cost_per_ton_km * per_factory_tons
        total_co2 += co2
        total_cost += cost

    kpis = {
        "baseline_total_units": float(total_units),
        "baseline_total_co2_kg": float(total_co2),
        "baseline_total_cost_usd": float(total_cost),
        "avg_co2_per_unit_kg": float(total_co2 / total_units),
    }
    return kpis


def plot_distributions(data_dir: str, out_dir: str):
    ensure_dir(out_dir)
    routes_sf = pd.read_csv(os.path.join(data_dir, "routes_sup_to_fac.csv"))
    routes_fr = pd.read_csv(os.path.join(data_dir, "routes_fac_to_reg.csv"))

    plt.figure(figsize=(8, 4))
    sns.histplot(routes_sf["distance_km"], bins=30, color="steelblue")
    plt.title("Supplier→Factory Route Distances (km)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distances_sf.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(routes_fr["distance_km"], bins=30, color="seagreen")
    plt.title("Factory→Region Route Distances (km)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distances_fr.png"))
    plt.close()

    # Emission factor by mode
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="mode", y="ef_kg_per_ton_km", data=routes_sf)
    plt.title("Emission factors by mode (S→F)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ef_by_mode_sf.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()
    ensure_dir(args.out_dir)

    kpis = compute_kpis(args.data_dir)
    with open(os.path.join(args.out_dir, "kpis.txt"), "w") as f:
        for k, v in kpis.items():
            f.write(f"{k}: {v}\n")
    print("Baseline KPIs written to outputs/kpis.txt")

    plot_distributions(args.data_dir, args.out_dir)
    print("EDA plots saved to outputs/")


if __name__ == "__main__":
    main()

