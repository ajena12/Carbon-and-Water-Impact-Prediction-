import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from faker import Faker

from cfo.utils.emissions import (
    haversine_km,
    DEFAULT_TRANSPORT_FACTORS,
    DEFAULT_TRANSPORT_COSTS,
)


np.random.seed(42)
fake = Faker()
Faker.seed(42)


@dataclass
class GeneratorConfig:
    num_suppliers: int = 15
    num_factories: int = 5
    num_regions: int = 10
    product_weight_kg: float = 0.5
    material_kg_per_unit: float = 0.4
    min_supplier_supply_tons: float = 200.0
    max_supplier_supply_tons: float = 800.0
    min_factory_capacity_units: int = 5000
    max_factory_capacity_units: int = 20000
    min_region_demand_units: int = 1500
    max_region_demand_units: int = 6000


EU_BOUNDS = {
    "lat_min": 36.0,  # approx southern Europe
    "lat_max": 60.0,  # approx northern Europe
    "lon_min": -10.0,  # west
    "lon_max": 30.0,   # east
}


def random_point_in_europe(n: int) -> np.ndarray:
    lats = np.random.uniform(EU_BOUNDS["lat_min"], EU_BOUNDS["lat_max"], size=n)
    lons = np.random.uniform(EU_BOUNDS["lon_min"], EU_BOUNDS["lon_max"], size=n)
    return np.vstack([lats, lons]).T


def generate_entities(cfg: GeneratorConfig):
    # Suppliers
    supplier_points = random_point_in_europe(cfg.num_suppliers)
    suppliers = pd.DataFrame(
        {
            "supplier_id": [f"S{i+1:03d}" for i in range(cfg.num_suppliers)],
            "name": [f"Supplier {i+1}" for i in range(cfg.num_suppliers)],
            "lat": supplier_points[:, 0],
            "lon": supplier_points[:, 1],
            "material_type": np.random.choice(["glass", "plastic", "paper", "pigment"], size=cfg.num_suppliers),
            "supply_capacity_tons": np.random.uniform(
                cfg.min_supplier_supply_tons, cfg.max_supplier_supply_tons, size=cfg.num_suppliers
            ).round(2),
        }
    )

    # Factories
    factory_points = random_point_in_europe(cfg.num_factories)
    factories = pd.DataFrame(
        {
            "factory_id": [f"F{i+1:02d}" for i in range(cfg.num_factories)],
            "name": [f"Factory {i+1}" for i in range(cfg.num_factories)],
            "lat": factory_points[:, 0],
            "lon": factory_points[:, 1],
            "capacity_units": np.random.randint(
                cfg.min_factory_capacity_units, cfg.max_factory_capacity_units + 1, size=cfg.num_factories
            ),
        }
    )

    # Regions (demand markets)
    region_points = random_point_in_europe(cfg.num_regions)
    regions = pd.DataFrame(
        {
            "region_id": [f"R{i+1:03d}" for i in range(cfg.num_regions)],
            "name": [fake.city() for _ in range(cfg.num_regions)],
            "lat": region_points[:, 0],
            "lon": region_points[:, 1],
        }
    )

    # Demand
    demand = pd.DataFrame(
        {
            "region_id": regions["region_id"],
            "product": "Beauty Product",
            "units_demanded": np.random.randint(
                cfg.min_region_demand_units, cfg.max_region_demand_units + 1, size=cfg.num_regions
            ),
        }
    )

    # Product
    product = pd.DataFrame(
        {
            "product_id": ["P001"],
            "name": ["Beauty Product"],
            "weight_kg_per_unit": [cfg.product_weight_kg],
            "material_kg_per_unit": [cfg.material_kg_per_unit],
        }
    )

    return suppliers, factories, regions, demand, product


def build_routes_sup_to_fac(suppliers: pd.DataFrame, factories: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    modes = list(DEFAULT_TRANSPORT_FACTORS.keys())
    for _, s in suppliers.iterrows():
        for _, f in factories.iterrows():
            distance_km = haversine_km(s.lat, s.lon, f.lat, f.lon)
            mode = np.random.choice(modes, p=[0.6, 0.2, 0.15, 0.05])  # mostly truck
            rows.append(
                {
                    "supplier_id": s.supplier_id,
                    "factory_id": f.factory_id,
                    "mode": mode,
                    "distance_km": round(distance_km, 2),
                    "ef_kg_per_ton_km": DEFAULT_TRANSPORT_FACTORS[mode],
                    "cost_per_ton_km": DEFAULT_TRANSPORT_COSTS[mode],
                }
            )
    return pd.DataFrame(rows)


def build_routes_fac_to_reg(factories: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    modes = list(DEFAULT_TRANSPORT_FACTORS.keys())
    for _, f in factories.iterrows():
        for _, r in regions.iterrows():
            distance_km = haversine_km(f.lat, f.lon, r.lat, r.lon)
            # Trucks dominate land, with some rail and ship (air rarely)
            mode = np.random.choice(modes, p=[0.7, 0.2, 0.08, 0.02])
            rows.append(
                {
                    "factory_id": f.factory_id,
                    "region_id": r.region_id,
                    "mode": mode,
                    "distance_km": round(distance_km, 2),
                    "ef_kg_per_ton_km": DEFAULT_TRANSPORT_FACTORS[mode],
                    "cost_per_ton_km": DEFAULT_TRANSPORT_COSTS[mode],
                }
            )
    return pd.DataFrame(rows)


def write_transport_modes(out_dir: str):
    df = pd.DataFrame(
        {
            "mode": list(DEFAULT_TRANSPORT_FACTORS.keys()),
            "ef_kg_per_ton_km": list(DEFAULT_TRANSPORT_FACTORS.values()),
            "cost_per_ton_km": [DEFAULT_TRANSPORT_COSTS[m] for m in DEFAULT_TRANSPORT_FACTORS.keys()],
        }
    )
    df.to_csv(os.path.join(out_dir, "transport_modes.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    cfg = GeneratorConfig()
    suppliers, factories, regions, demand, product = generate_entities(cfg)

    routes_sf = build_routes_sup_to_fac(suppliers, factories)
    routes_fr = build_routes_fac_to_reg(factories, regions)

    suppliers.to_csv(os.path.join(out_dir, "suppliers.csv"), index=False)
    factories.to_csv(os.path.join(out_dir, "factories.csv"), index=False)
    regions.to_csv(os.path.join(out_dir, "regions.csv"), index=False)
    demand.to_csv(os.path.join(out_dir, "demand.csv"), index=False)
    product.to_csv(os.path.join(out_dir, "product.csv"), index=False)
    routes_sf.to_csv(os.path.join(out_dir, "routes_sup_to_fac.csv"), index=False)
    routes_fr.to_csv(os.path.join(out_dir, "routes_fac_to_reg.csv"), index=False)
    write_transport_modes(out_dir)

    print(f"Data generated in {out_dir}")


if __name__ == "__main__":
    main()

