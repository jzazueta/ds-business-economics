"""Generate the Chapter 14 dataset: store x SKU demand for a Saltillo grocery chain.

Outputs (in ../data/):
    ch14_stores.csv  one row per store (observed store attributes)
    ch14_skus.csv    one row per SKU (observed product attributes)
    ch14_sales.csv   one row per store-SKU pair the store carries
                     (average weekly units over the last 13 weeks)

Demand has three parts: a store effect, a SKU effect, and a "taste"
interaction between hidden store and product traits. The hidden traits are
only partly explained by the observed attributes, which is what gives
learned embeddings an advantage on store-SKU pairs never seen in training.

Run:  python ch14_generate_dataset.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 14
N_STORES = 300
N_SKUS = 400
N_WEEKS = 13
K = 4                 # hidden taste dimensions
TASTE_SCALE = 0.35    # strength of the store x SKU interaction
PAIR_NOISE = 0.45     # store-SKU idiosyncratic noise (log scale)

OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ── Stores ────────────────────────────────────────────────────────────────
# City, state, and whether the city sits on the US border.
cities = pd.DataFrame(
    [
        ("Saltillo", "Coahuila", 0, 70),
        ("Ramos Arizpe", "Coahuila", 0, 12),
        ("Monclova", "Coahuila", 0, 22),
        ("Torreón", "Coahuila", 0, 45),
        ("Piedras Negras", "Coahuila", 1, 18),
        ("Acuña", "Coahuila", 1, 12),
        ("Monterrey", "Nuevo León", 0, 60),
        ("Nuevo Laredo", "Tamaulipas", 1, 25),
        ("Gómez Palacio", "Durango", 0, 20),
        ("Zacatecas", "Zacatecas", 0, 16),
    ],
    columns=["city", "state", "border", "n"],
)
assert cities.n.sum() == N_STORES
city_idx = np.repeat(np.arange(len(cities)), cities.n)
rng.shuffle(city_idx)

fmt = rng.choice(["express", "super", "hiper"], N_STORES, p=[0.40, 0.45, 0.15])
floor_m2 = np.where(
    fmt == "express", rng.normal(350, 60, N_STORES),
    np.where(fmt == "super", rng.normal(1800, 300, N_STORES), rng.normal(5500, 800, N_STORES)),
).round(-1)
# Neighborhood socioeconomic index (z-score); higher = more affluent
nse_z = rng.normal(0, 1, N_STORES)
nse = pd.cut(nse_z, [-np.inf, -0.8, 0.2, 1.0, np.inf], labels=["D", "C", "C+", "AB"]).astype(str)
border = cities.border.values[city_idx]

# Hidden store tastes. Each dimension is partly tied to something observable
# (affluence, the border, store size) and partly invisible to the analyst.
U = rng.normal(0, 1, (N_STORES, K))
U[:, 0] = 0.5 * nse_z + np.sqrt(0.75) * rng.normal(0, 1, N_STORES)          # premium
border_z = (border - border.mean()) / border.std()
U[:, 1] = 0.9 * border_z + np.sqrt(0.19) * rng.normal(0, 1, N_STORES)       # US brands, strong at the border
U[:, 2] = 0.4 * (fmt == "hiper") - 0.4 * (fmt == "express") + rng.normal(0, 0.95, N_STORES)  # bulk/family
# U[:, 3] stays fully hidden: a "healthy eating" lean with no observed proxy

store_effect = (
    rng.normal(0, 0.4, N_STORES)
    + pd.Series(fmt).map({"express": -0.4, "super": 0.0, "hiper": 0.5}).values
)

stores = pd.DataFrame(
    {
        "store_id": [f"S{i:03d}" for i in range(N_STORES)],
        "city": cities.city.values[city_idx],
        "state": cities.state.values[city_idx],
        "format": fmt,
        "floor_m2": floor_m2.astype(int),
        "nse": nse,
    }
)

# ── SKUs ──────────────────────────────────────────────────────────────────
categories = ["dairy", "snacks", "beverages", "pantry", "cleaning",
              "personal_care", "bakery", "frozen"]
cat_price = dict(dairy=32, snacks=24, beverages=28, pantry=35, cleaning=48,
                 personal_care=62, bakery=30, frozen=70)
cat = rng.integers(0, len(categories), N_SKUS)
tier = rng.choice(["national", "private_label", "imported"], N_SKUS, p=[0.6, 0.25, 0.15])
tier_mult = pd.Series(tier).map({"national": 1.0, "private_label": 0.75, "imported": 1.6}).values
price = (np.array([cat_price[categories[c]] for c in cat]) * tier_mult
         * np.exp(rng.normal(0, 0.35, N_SKUS))).round(1)
pack = rng.choice(["single", "multi"], N_SKUS, p=[0.7, 0.3])

# Hidden product traits, loading on the same four taste dimensions.
cat_load = rng.normal(0, 0.6, (len(categories), K))
V = cat_load[cat] + rng.normal(0, 0.8, (N_SKUS, K))
V[:, 0] += 0.6 * (tier == "imported") - 0.5 * (tier == "private_label")
V[:, 1] += 1.0 * (tier == "imported")
V[:, 2] += 0.5 * (pack == "multi")

sku_effect = rng.normal(0, 0.5, N_SKUS) - 0.3 * np.log(price / 20)

skus = pd.DataFrame(
    {
        "sku_id": [f"P{j:03d}" for j in range(N_SKUS)],
        "category": [categories[c] for c in cat],
        "brand_tier": tier,
        "pack": pack,
        "price_mxn": price,
    }
)

# ── Assortment and sales ──────────────────────────────────────────────────
carry_p = pd.Series(fmt).map({"express": 0.15, "super": 0.30, "hiper": 0.50}).values
carry = rng.random((N_STORES, N_SKUS)) < carry_p[:, None]
s, p = np.where(carry)

log_mu = (1.8 + store_effect[s] + sku_effect[p]
          + TASTE_SCALE * (U[s] * V[p]).sum(axis=1)
          + rng.normal(0, PAIR_NOISE, len(s)))
weekly = rng.poisson(np.exp(log_mu)[:, None], (len(s), N_WEEKS))

sales = pd.DataFrame(
    {
        "store_id": stores.store_id.values[s],
        "sku_id": skus.sku_id.values[p],
        "units_13wk": weekly.sum(axis=1),
    }
)
sales["units_per_week"] = (sales.units_13wk / N_WEEKS).round(2)

stores.to_csv(OUT / "ch14_stores.csv", index=False)
skus.to_csv(OUT / "ch14_skus.csv", index=False)
sales.to_csv(OUT / "ch14_sales.csv", index=False)
print(f"stores {len(stores)}, skus {len(skus)}, carried pairs {len(sales)} "
      f"({len(sales) / (N_STORES * N_SKUS):.1%} of all pairs)")
