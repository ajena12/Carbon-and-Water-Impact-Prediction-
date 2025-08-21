import math
from typing import Dict


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two lat/lon pairs in kilometers."""
    radius_earth_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius_earth_km * c


DEFAULT_TRANSPORT_FACTORS: Dict[str, float] = {
    # kg CO2 per ton-km (illustrative values)
    "truck": 0.1,
    "rail": 0.03,
    "ship": 0.015,
    "air": 0.6,
}


DEFAULT_TRANSPORT_COSTS: Dict[str, float] = {
    # USD per ton-km (illustrative)
    "truck": 0.09,
    "rail": 0.05,
    "ship": 0.03,
    "air": 0.8,
}


def estimate_emissions_kg(distance_km: float, load_tons: float, ef_kg_per_ton_km: float) -> float:
    if distance_km <= 0 or load_tons <= 0 or ef_kg_per_ton_km <= 0:
        return 0.0
    return distance_km * load_tons * ef_kg_per_ton_km


def estimate_cost_usd(distance_km: float, load_tons: float, cost_per_ton_km: float) -> float:
    if distance_km <= 0 or load_tons <= 0 or cost_per_ton_km <= 0:
        return 0.0
    return distance_km * load_tons * cost_per_ton_km
