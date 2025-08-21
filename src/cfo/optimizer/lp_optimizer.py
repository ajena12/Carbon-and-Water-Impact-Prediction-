import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import pulp as pl


def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    suppliers = pd.read_csv(os.path.join(data_dir, "suppliers.csv"))
    factories = pd.read_csv(os.path.join(data_dir, "factories.csv"))
    regions = pd.read_csv(os.path.join(data_dir, "regions.csv"))
    demand = pd.read_csv(os.path.join(data_dir, "demand.csv"))
    routes_sf = pd.read_csv(os.path.join(data_dir, "routes_sup_to_fac.csv"))
    routes_fr = pd.read_csv(os.path.join(data_dir, "routes_fac_to_reg.csv"))
    return suppliers, factories, regions, demand, routes_sf, routes_fr


def optimize_min_co2(data_dir: str, results_path: str, cost_budget_multiplier: float = 1.0) -> pd.DataFrame:
    suppliers, factories, regions, demand, routes_sf, routes_fr = load_data(data_dir)

    # Convert units: material tons for supply, product units for demand; link via material per unit
    product = pd.read_csv(os.path.join(data_dir, "product.csv"))
    material_kg_per_unit = float(product["material_kg_per_unit"].iloc[0])
    weight_kg_per_unit = float(product["weight_kg_per_unit"].iloc[0])

    # Decision variables: flow of material S->F in tons; flow of product F->R in units
    prob = pl.LpProblem("Minimize_CO2", pl.LpMinimize)

    # Create keys for routes
    sf_keys = [(r.supplier_id, r.factory_id) for _, r in routes_sf.iterrows()]
    fr_keys = [(r.factory_id, r.region_id) for _, r in routes_fr.iterrows()]

    x_sf = pl.LpVariable.dicts("flow_sf_tons", sf_keys, lowBound=0, cat="Continuous")
    y_fr = pl.LpVariable.dicts("flow_fr_units", fr_keys, lowBound=0, cat="Continuous")

    # Objective: Minimize total CO2
    co2_sf = {
        (r.supplier_id, r.factory_id): r.distance_km * r.ef_kg_per_ton_km for _, r in routes_sf.iterrows()
    }
    # For F->R, convert units to tons using product weight
    co2_fr_per_unit = {
        (r.factory_id, r.region_id): r.distance_km * r.ef_kg_per_ton_km * (weight_kg_per_unit / 1000.0)
        for _, r in routes_fr.iterrows()
    }

    prob += (
        pl.lpSum(co2_sf[k] * x_sf[k] for k in sf_keys)
        + pl.lpSum(co2_fr_per_unit[k] * y_fr[k] for k in fr_keys)
    ), "Total_CO2_kg"

    # Constraints
    # 1) Supplier material capacity (tons)
    for _, s in suppliers.iterrows():
        outgoing = [x_sf[(s.supplier_id, fid)] for fid in factories.factory_id]
        prob += pl.lpSum(outgoing) <= s.supply_capacity_tons, f"SupplierCap_{s.supplier_id}"

    # 2) Factory material balance: incoming material in tons must be enough for produced units
    for _, f in factories.iterrows():
        incoming_tons = pl.lpSum(x_sf[(sid, f.factory_id)] for sid in suppliers.supplier_id)
        outgoing_units = pl.lpSum(y_fr[(f.factory_id, rid)] for rid in regions.region_id)
        prob += incoming_tons >= (outgoing_units * material_kg_per_unit / 1000.0), f"MaterialBalance_{f.factory_id}"

    # 3) Factory production capacity (units)
    for _, f in factories.iterrows():
        outgoing_units = pl.lpSum(y_fr[(f.factory_id, rid)] for rid in regions.region_id)
        prob += outgoing_units <= f.capacity_units, f"FactoryCap_{f.factory_id}"

    # 4) Demand satisfaction (units)
    demand_map = dict(zip(demand.region_id, demand.units_demanded))
    for _, r in regions.iterrows():
        incoming_units = pl.lpSum(y_fr[(fid, r.region_id)] for fid in factories.factory_id)
        prob += incoming_units >= demand_map[r.region_id], f"Demand_{r.region_id}"

    # 5) Optional budget constraint: do not exceed baseline cost * multiplier
    # Compute costs
    cost_sf = {
        (r.supplier_id, r.factory_id): r.distance_km * r.cost_per_ton_km for _, r in routes_sf.iterrows()
    }
    cost_fr_per_unit = {
        (r.factory_id, r.region_id): r.distance_km * r.cost_per_ton_km * (weight_kg_per_unit / 1000.0)
        for _, r in routes_fr.iterrows()
    }

    # Baseline naive allocation: split evenly across shortest routes per pair to estimate a cost ceiling
    # For simplicity, use greedy nearest-factory for each region and nearest-factory for each supplier
    baseline_cost = 0.0
    # Regions: choose nearest factory
    fr_sorted = routes_fr.sort_values(["region_id", "distance_km"]).groupby("region_id").head(1)
    demand_units_total = demand.units_demanded.sum()
    fr_share = { (r.factory_id, r.region_id): demand_map[r.region_id] for _, r in fr_sorted.iterrows() }
    baseline_cost += sum(cost_fr_per_unit[k] * v for k, v in fr_share.items())
    # Suppliers: allocate material proportionally to factory output needs
    required_material_tons = (demand_units_total * material_kg_per_unit) / 1000.0
    sf_sorted = routes_sf.sort_values(["factory_id", "distance_km"]).groupby("factory_id").head(1)
    # split required material evenly across chosen suppliers for each factory (approx)
    sf_share = {}
    per_factory_units = demand_units_total / len(factories)
    per_factory_tons = (per_factory_units * material_kg_per_unit) / 1000.0
    for _, r in sf_sorted.iterrows():
        sf_share[(r.supplier_id, r.factory_id)] = per_factory_tons
    baseline_cost += sum(cost_sf[k] * v for k, v in sf_share.items())

    prob += (
        pl.lpSum(cost_sf[k] * x_sf[k] for k in sf_keys)
        + pl.lpSum(cost_fr_per_unit[k] * y_fr[k] for k in fr_keys)
    ) <= baseline_cost * cost_budget_multiplier, "Budget"

    # Solve
    prob.solve(pl.PULP_CBC_CMD(msg=False))

    status = pl.LpStatus[prob.status]
    if status != "Optimal":
        print(f"Warning: solver status {status}")

    rows = []
    # Extract flows and KPIs
    total_co2 = pl.value(prob.objective)
    total_cost = (
        sum(cost_sf[k] * x_sf[k].value() for k in sf_keys)
        + sum(cost_fr_per_unit[k] * y_fr[k].value() for k in fr_keys)
    )

    for k in sf_keys:
        flow = x_sf[k].value() or 0.0
        if flow > 1e-6:
            rows.append({
                "from_type": "supplier",
                "from_id": k[0],
                "to_type": "factory",
                "to_id": k[1],
                "flow_tons": flow,
                "flow_units": np.nan,
            })
    for k in fr_keys:
        flow = y_fr[k].value() or 0.0
        if flow > 1e-6:
            rows.append({
                "from_type": "factory",
                "from_id": k[0],
                "to_type": "region",
                "to_id": k[1],
                "flow_tons": np.nan,
                "flow_units": flow,
            })

    df = pd.DataFrame(rows)
    df.attrs["total_co2_kg"] = total_co2
    df.attrs["total_cost_usd"] = total_cost
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)
    print(f"Optimization results saved to {results_path}. CO2_kg={total_co2:.2f}, Cost=${total_cost:,.2f}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--results_path", type=str, default="outputs/optimization_results.csv")
    parser.add_argument("--cost_budget_multiplier", type=float, default=1.0)
    args = parser.parse_args()
    optimize_min_co2(args.data_dir, args.results_path, args.cost_budget_multiplier)


if __name__ == "__main__":
    main()

