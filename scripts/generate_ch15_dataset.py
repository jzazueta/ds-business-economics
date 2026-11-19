"""Generate the synthetic site-selection data for Chapter 15.

Horno del Valle, a bakery-cafe chain in San Luis Potosi and the Bajio,
has 60 stores and a shortlist of 300 candidate sites. A store's sales
depend on its neighborhood, and fall when another of the chain's own
stores sits nearby: the two split customers, and the split fades with
distance.

Outputs:
    data/ch15_stores.csv      60 existing stores with annual sales
    data/ch15_candidates.csv  the shortlist as delivered: 300 sites in 312
                              rows, with duplicates, mixed city spellings,
                              income as text and one proposer's foot
                              traffic on a 0-1 scale (cleaned in Sec. 15.2)
"""

import numpy as np
import pandas as pd

SEED = 1515
rng = np.random.default_rng(SEED)

# Cannibalization: each own store within reach removes a share of sales
# that decays with distance (km).
CANNIBAL_SHARE = 0.30
CANNIBAL_SCALE_KM = 1.2

CITIES = {
    # name: (lat, lon, n_stores, n_candidates, pop_scale)
    "San Luis Potosi": (22.1510, -100.9760, 18, 75, 1.00),
    "Queretaro":       (20.5930, -100.3900, 16, 70, 1.05),
    "Leon":            (21.1220, -101.6820, 14, 65, 1.10),
    "Celaya":          (20.5230, -100.8150, 6,  25, 0.70),
    "Irapuato":        (20.6770, -101.3560, 6,  25, 0.70),
}
N_HIGHWAY = 40  # candidate sites on highways and industrial parks


def to_latlon(lat0, lon0, dx_km, dy_km):
    """Convert km offsets from a city center into latitude and longitude."""
    lat = lat0 + dy_km / 111.0
    lon = lon0 + dx_km / (111.0 * np.cos(np.radians(lat0)))
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def urban_cells(city, lat0, lon0, n, pop_scale):
    """Draw n urban locations with neighborhood features."""
    r = rng.gamma(2.0, 2.2, n)                      # km from center
    theta = rng.uniform(0, 2 * np.pi, n)
    dx, dy = r * np.cos(theta), r * np.sin(theta)
    lat, lon = to_latlon(lat0, lon0, dx, dy)

    pop = pop_scale * 42000 * np.exp(-r / 7.0) * rng.lognormal(0, 0.25, n)
    # One affluent sector per city (an arc of directions)
    affluent_dir = rng.uniform(0, 2 * np.pi)
    arc = np.cos(theta - affluent_dir)
    income = 11000 + 9000 * np.clip(arc, 0, None) * np.exp(-r / 10) \
        + rng.normal(0, 1500, n)
    competitors = rng.poisson(pop / 3500)
    traffic = np.clip(35 + 40 * np.exp(-r / 4) + rng.normal(0, 10, n), 5, 100)
    return pd.DataFrame({
        "city": city, "lat": lat, "lon": lon,
        "population_2km": pop.round(-1).astype(int),
        "median_income": income.round(-1).astype(int),
        "competitors_2km": competitors,
        "foot_traffic": traffic.round(1),
        "site_type": "urban",
    })


def highway_cells(n):
    """Highway and industrial-park sites between cities: few residents,
    heavy traffic, middling income."""
    names = list(CITIES)
    rows = []
    for _ in range(n):
        a, b = rng.choice(len(names), 2, replace=False)
        (la, loa, *_), (lb, lob, *_) = CITIES[names[a]], CITIES[names[b]]
        t = rng.uniform(0.15, 0.85)
        rows.append({
            "city": "Highway",
            "lat": la + t * (lb - la) + rng.normal(0, 0.01),
            "lon": loa + t * (lob - loa) + rng.normal(0, 0.01),
            "population_2km": int(round(rng.uniform(1500, 6000), -1)),
            "median_income": int(round(rng.normal(12500, 1200), -1)),
            "competitors_2km": int(rng.poisson(1.2)),
            "foot_traffic": round(float(np.clip(rng.normal(82, 7), 50, 100)), 1),
            "site_type": "highway",
        })
    return pd.DataFrame(rows)


def potential_sales(df):
    """Annual sales (MXN thousands) a store would make with no own-store
    neighbors."""
    log_s = (np.log(2600)
             + 0.30 * np.log(df["population_2km"] / 20000)
             + 0.85 * np.log(df["median_income"] / 13000)
             - 0.03 * df["competitors_2km"]
             + 0.010 * (df["foot_traffic"] - 60))
    # Highway sites: travelers buy less per visit than residents.
    log_s = log_s - 0.25 * (df["site_type"] == "highway")
    return np.exp(log_s)


# ---------------------------------------------------------------------------
# 1. Urban pool per city; existing stores chosen from the best of it
# ---------------------------------------------------------------------------
stores, candidates = [], []
for city, (lat0, lon0, n_st, n_cand, ps) in CITIES.items():
    pool = urban_cells(city, lat0, lon0, 6 * (n_st + n_cand), ps)
    pot = potential_sales(pool)
    # The chain opened where potential was high, which places stores
    # close to one another in the affluent sector.
    w = pot ** 2
    st_idx = rng.choice(len(pool), n_st, replace=False, p=w / w.sum())
    stores.append(pool.iloc[st_idx])
    rest = pool.drop(pool.index[st_idx])
    # Candidates: a mix that leans toward good sites, as a shortlist would
    w2 = potential_sales(rest) ** 1.5
    cand_idx = rng.choice(len(rest), n_cand, replace=False, p=w2 / w2.sum())
    candidates.append(rest.iloc[cand_idx])

stores = pd.concat(stores, ignore_index=True)
candidates = pd.concat(candidates + [highway_cells(N_HIGHWAY)],
                       ignore_index=True)

# ---------------------------------------------------------------------------
# 2. Sales with cannibalization among existing stores
# ---------------------------------------------------------------------------
D = haversine_km(stores["lat"].values[:, None], stores["lon"].values[:, None],
                 stores["lat"].values[None, :], stores["lon"].values[None, :])
np.fill_diagonal(D, np.inf)
retain = np.prod(1 - CANNIBAL_SHARE * np.exp(-D / CANNIBAL_SCALE_KM), axis=1)
noise = rng.lognormal(0, 0.12, len(stores))
stores["annual_sales"] = (potential_sales(stores) * retain * noise).round(0)

stores["dist_nearest_own_km"] = D.min(axis=1).round(2)
stores["own_within_2km"] = (D < 2).sum(axis=1)

# ---------------------------------------------------------------------------
# 3. Two growth-area sites next to an isolated store: strong enough that
#    they earn a place even after the sales they take from that store
# ---------------------------------------------------------------------------
isolated = stores.index[stores["dist_nearest_own_km"] > 4.0]
planted_rows = []
for st in rng.choice(isolated, 2, replace=False):
    lat0, lon0 = stores.loc[st, ["lat", "lon"]]
    while True:
        ang = rng.uniform(0, 2 * np.pi)
        lat, lon = to_latlon(lat0, lon0, 1.6 * np.cos(ang), 1.6 * np.sin(ang))
        d = haversine_km(lat, lon, stores["lat"].values, stores["lon"].values)
        if (d < 2).sum() == 1:
            break
    planted_rows.append({
        "city": stores.loc[st, "city"], "lat": lat, "lon": lon,
        "population_2km": int(round(rng.uniform(34000, 38000), -1)),
        "median_income": int(round(rng.uniform(18500, 19500), -1)),
        "competitors_2km": 2,
        "foot_traffic": round(float(rng.uniform(70, 76)), 1),
        "site_type": "urban",
    })
# Replace two ordinary urban candidates from the same cities, keeping 300
for k, row in enumerate(planted_rows):
    idx = candidates.index[(candidates["city"] == row["city"])
                           & (candidates["site_type"] == "urban")][-1 - k]
    candidates.loc[idx, list(row)] = list(row.values())

Dc = haversine_km(candidates["lat"].values[:, None],
                  candidates["lon"].values[:, None],
                  stores["lat"].values[None, :], stores["lon"].values[None, :])
candidates["dist_nearest_own_km"] = Dc.min(axis=1).round(2)
candidates["own_within_2km"] = (Dc < 2).sum(axis=1)

# IDs, rounding, column order
stores.insert(0, "store_id", [f"HV{i+1:03d}" for i in range(len(stores))])
candidates.insert(0, "site_id", [f"C{i+1:03d}" for i in range(len(candidates))])
for df in (stores, candidates):
    df["lat"] = df["lat"].round(5)
    df["lon"] = df["lon"].round(5)

stores.drop(columns="site_type").to_csv("data/ch15_stores.csv", index=False)
candidates = candidates.drop(columns="site_type")

# ---------------------------------------------------------------------------
# 4. The shortlist as the real estate team delivers it: four people's
#    spreadsheets stacked together, each with its own habits
# ---------------------------------------------------------------------------
mess = np.random.default_rng(SEED + 1)   # separate stream: steps 1-3 unchanged
PEOPLE = ["Arturo", "Beatriz", "Carmen", "Diego"]
candidates["proposed_by"] = mess.choice(PEOPLE, len(candidates))

# Twelve sites proposed twice, by a second person, with a new ID.
# C074 is always one of them: it heads the ranking, so the duplicate shows.
others = candidates.index[candidates["site_id"] != "C074"]
dup_idx = np.r_[candidates.index[candidates["site_id"] == "C074"],
                mess.choice(others, 11, replace=False)]
dups = candidates.loc[dup_idx].copy()
dups["proposed_by"] = [mess.choice([p for p in PEOPLE if p != q])
                       for q in dups["proposed_by"]]
dups["site_id"] = [f"C{301 + k}" for k in range(len(dups))]
raw = pd.concat([candidates, dups], ignore_index=True)

# Each person's way of writing city names
STYLE = {
    "Beatriz": {"San Luis Potosi": "SLP", "Queretaro": "Qro.",
                "Leon": "Leon, Gto.", "Celaya": "Celaya, Gto.",
                "Irapuato": "Irapuato, Gto.", "Highway": "Carretera"},
    "Carmen":  {c: c.upper() for c in list(CITIES) + ["Highway"]},
    "Diego":   {c: c.lower() + " " for c in list(CITIES) + ["Highway"]},
}
for person, names in STYLE.items():
    rows = raw["proposed_by"] == person
    raw.loc[rows, "city"] = raw.loc[rows, "city"].map(names)

# Carmen records foot traffic as a share (0-1), not a score (0-100)
rows = raw["proposed_by"] == "Carmen"
raw["foot_traffic"] = raw["foot_traffic"].astype(object)
raw.loc[rows, "foot_traffic"] = (raw.loc[rows, "foot_traffic"]
                                 .astype(float) / 100).round(3)
# Beatriz types income as currency text: "$17,330"
raw["median_income"] = raw["median_income"].astype(object)
rows = raw["proposed_by"] == "Beatriz"
raw.loc[rows, "median_income"] = [f"${v:,}" for v in
                                  raw.loc[rows, "median_income"]]

# Stack the spreadsheets person by person
raw = raw.sort_values(["proposed_by", "site_id"]).reset_index(drop=True)
raw.to_csv("data/ch15_candidates.csv", index=False)
print(stores.shape, raw.shape)
