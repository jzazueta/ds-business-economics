"""Fetch trade-structure indicators from the World Bank WDI API (Chapter 11).

Run in Colab:  !python fetch_wdi_trade.py
Outputs (in the working directory):
  ch11_wdi_raw_long.csv   every observation returned, 2015-2024, for provenance
  ch11_wdi_snapshot.csv   one row per country, latest non-missing value per indicator
"""
import datetime
import time

import pandas as pd
import requests

BASE = "https://api.worldbank.org/v2"
YEARS = "2015:2024"

# Indicator code -> short column name used in the chapter
INDICATORS = {
    # Composition of merchandise exports (% of merchandise exports)
    "TX.VAL.FOOD.ZS.UN": "exp_food",
    "TX.VAL.AGRI.ZS.UN": "exp_agri_raw",
    "TX.VAL.FUEL.ZS.UN": "exp_fuel",
    "TX.VAL.MMTL.ZS.UN": "exp_ores_metals",
    "TX.VAL.MANF.ZS.UN": "exp_manuf",
    # High-technology exports (% of manufactured exports)
    "TX.VAL.TECH.MF.ZS": "hightech_share",
    # Composition of merchandise imports (% of merchandise imports)
    "TM.VAL.FUEL.ZS.UN": "imp_fuel",
    "TM.VAL.MANF.ZS.UN": "imp_manuf",
    # Openness and scale
    "NE.EXP.GNFS.ZS": "exports_gdp",
    "NE.IMP.GNFS.ZS": "imports_gdp",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_gdp",
    "NY.GDP.PCAP.CD": "gdp_pc_usd",
    "NY.GDP.MKTP.CD": "gdp_usd",
    "SP.POP.TOTL": "population",
}


def get_json(url, params, retries=3):
    """GET a World Bank endpoint and return the parsed JSON, retrying on failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as err:  # network hiccups are common on long pulls
            print(f"  retry {attempt + 1} after error: {err}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def get_countries():
    """Country metadata, excluding regional and income aggregates."""
    data = get_json(f"{BASE}/country", {"format": "json", "per_page": 400})
    rows = [
        {
            "iso3": c["id"],
            "country": c["name"],
            "region": c["region"]["value"],
            "income_group": c["incomeLevel"]["value"],
        }
        for c in data[1]
        if c["region"]["value"] != "Aggregates"
    ]
    return pd.DataFrame(rows)


def get_indicator(code):
    """All country-year observations for one indicator, as a long table."""
    rows, page, pages = [], 1, 1
    while page <= pages:
        meta, obs = get_json(
            f"{BASE}/country/all/indicator/{code}",
            {"format": "json", "per_page": 20000, "date": YEARS, "page": page},
        )
        pages = meta["pages"]
        for o in obs or []:
            rows.append(
                {
                    "iso3": o["countryiso3code"],
                    "indicator": code,
                    "year": int(o["date"]),
                    "value": o["value"],
                }
            )
        page += 1
    return pd.DataFrame(rows)


def main():
    countries = get_countries()
    print(f"Countries (aggregates removed): {len(countries)}")

    frames = []
    for code, name in INDICATORS.items():
        df = get_indicator(code)
        print(f"{name:16s} {code:22s} {df['value'].notna().sum():5d} non-missing obs")
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long = long[long["iso3"].isin(countries["iso3"])]
    long.to_csv("ch11_wdi_raw_long.csv", index=False)

    # Latest non-missing value per country and indicator, plus the year it came from
    latest = (
        long.dropna(subset=["value"])
        .sort_values("year")
        .groupby(["iso3", "indicator"])
        .last()
        .reset_index()
    )
    values = latest.pivot(index="iso3", columns="indicator", values="value")
    years = latest.pivot(index="iso3", columns="indicator", values="year")
    values = values.rename(columns=INDICATORS)
    years = years.rename(columns={k: f"yr_{v}" for k, v in INDICATORS.items()})

    snap = countries.merge(values, on="iso3", how="left").merge(years, on="iso3", how="left")
    snap["fetched_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    snap.to_csv("ch11_wdi_snapshot.csv", index=False)

    # Quick coverage report
    cols = list(INDICATORS.values())
    big = snap[snap["population"] >= 1e6]
    print(f"\nSnapshot rows: {len(snap)}; with population >= 1M: {len(big)}")
    print("Missing share among population >= 1M:")
    print(big[cols].isna().mean().round(3).to_string())
    print(f"Complete cases (population >= 1M): {big[cols].dropna().shape[0]}")


if __name__ == "__main__":
    main()
