"""Build the Chapter 12 data snapshot from raw INEGI downloads.

Readers do not need to run this: the notebook loads the snapshot files in data/ and geo/.
It is kept so every number in the chapter can be regenerated from the public sources:

  DENUE 05_2026, retail classes 46112-46311 (minisupers are SCIAN 462112)
  Censo de Poblacion y Vivienda 2020, ITER (municipal population totals)
  Marco Geoestadistico, June 2017 (state and municipal boundaries, 204 MB)
  EMEC 2018 base, state-level monthly indices (retail sector 46)

Usage:  python scripts/build_ch12_snapshot.py   (downloads to raw/, about 400 MB on first run)
"""
import glob, json, os, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import requests

RAW, DATA, GEO = Path("raw"), Path("data"), Path("geo")
for d in (RAW, DATA, GEO):
    d.mkdir(exist_ok=True)

INEGI = "https://www.inegi.org.mx/contenidos"
SOURCES = {
    "denue": f"{INEGI}/masiva/denue/denue_00_46112-46311_csv.zip",
    "iter": f"{INEGI}/programas/ccpv/2020/datosabiertos/iter/iter_00_cpv2020_csv.zip",
    "marco": f"{INEGI}/productos/prod_serv/contenidos/espanol/bvinegi/productos/geografia/marcogeo/889463142683_s.zip",
    "emec": f"{INEGI}/programas/emec/2018/datosabiertos/emec_mensual_entidad_federativa_csv.zip",
}

def fetch(name):
    """Download a source once; resume a partial file; skip a complete one."""
    url = SOURCES[name]
    path = RAW / Path(url).name
    size = int(requests.head(url, timeout=60).headers.get("Content-Length", 0))
    while not path.exists() or path.stat().st_size < size:
        have = path.stat().st_size if path.exists() else 0
        with requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=120) as r:
            r.raise_for_status()
            mode = "ab" if r.status_code == 206 else "wb"      # 206 = server honoured the resume
            with open(path, mode) as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return path

def unzip(name, members=None):
    out = RAW / name
    if not out.exists():
        with zipfile.ZipFile(fetch(name)) as z:
            z.extractall(out, members=members)
    return out

# ── 1. Municipal population, 2020 census ─────────────────────────────────────
iter_dir = unzip("iter")
iter_csv = glob.glob(str(iter_dir / "**/conjunto_de_datos/*.csv"), recursive=True)[0]
iter_all = pd.read_csv(iter_csv, dtype=str,
                       usecols=["ENTIDAD", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "LONGITUD", "LATITUD", "POBTOT"])
mun_pop = iter_all[(iter_all.LOC == "0000") & (iter_all.MUN != "000")].copy()
mun_pop.drop(columns=["LONGITUD", "LATITUD"]).to_csv(DATA / "mun_pop_2020.csv", index=False)

# ── 2. Boundaries (2017 framework) ────────────────────────────────────────────
marco = RAW / "marco"
if not marco.exists():
    with zipfile.ZipFile(fetch("marco")) as z:
        z.extractall(marco, members=[m for m in z.namelist()
                                     if "areas_geoestadisticas_estatales." in m or "areas_geoestadisticas_municipales." in m])
states = gpd.read_file(glob.glob(str(marco / "**/areas_geoestadisticas_estatales.shp"), recursive=True)[0])
munis = gpd.read_file(glob.glob(str(marco / "**/areas_geoestadisticas_municipales.shp"), recursive=True)[0])
munis["key"] = munis.CVE_ENT + munis.CVE_MUN

# ── 3. DENUE minisupers and chain classification ─────────────────────────────
denue_dir = unzip("denue")
denue_csv = glob.glob(str(denue_dir / "**/conjunto_de_datos/*.csv"), recursive=True)[0]
denue = pd.read_csv(denue_csv, encoding="latin1", dtype=str, low_memory=False)
mini = denue[denue.codigo_act == "462112"].copy()
name, legal = mini.nom_estab.str.upper().fillna(""), mini.raz_social.str.upper().fillna("")
mini["chain"] = "Other"
for chain, pattern, col in [("BA Express", "NUEVA WAL MART", legal), ("Circle K", "TIENDAS EXTRA", legal),
                            ("Neto", "TIENDAS SUPER PRECIO", legal), ("7-Eleven", "7-ELEVEN", legal),
                            ("3B", "TIENDAS TRES B", legal), ("OXXO", r"\bOXXO\b", name)]:
    mini.loc[col.str.contains(pattern, regex=True), "chain"] = chain
REGIONS = {"Norte": [2, 5, 8, 19, 26, 28], "Centro Norte": [1, 3, 6, 10, 14, 16, 18, 24, 25, 32],
           "Centro": [9, 11, 13, 15, 17, 21, 22, 29], "Sur": [4, 7, 12, 20, 23, 27, 30, 31]}   # Banxico regions
ent_to_region = {f"{e:02d}": r for r, ents in REGIONS.items() for e in ents}
mini["cve_ent"] = mini.cve_ent.str.zfill(2)
mini["region"] = mini.cve_ent.map(ent_to_region)
mini["key"] = mini.cve_ent + mini.cve_mun.str.zfill(3)

# ── 4. Crosswalk: municipalities created after 2017 -> their 2017 parent ──────
def parent_by_points(lon, lat, weights, child_keys):
    """Assign each child municipality to the 2017 polygon holding most of its weight."""
    pts = gpd.GeoDataFrame({"child": child_keys, "w": weights},
                           geometry=gpd.points_from_xy(lon, lat), crs=4326).to_crs(munis.crs)
    joined = gpd.sjoin(pts, munis[["key", "geometry"]], how="inner")
    best = joined.groupby(["child", "key"]).w.sum().reset_index().sort_values("w").drop_duplicates("child", keep="last")
    return dict(zip(best.child, best.key))

def dms(s):
    m = re.match(r"\s*(\d+)°(\d+)'([\d.]+)\"\s*([NSEW])", str(s))
    if not m:
        return np.nan
    v = int(m[1]) + int(m[2]) / 60 + float(m[3]) / 3600
    return -v if m[4] in "SW" else v

mun_pop["key"] = mun_pop.ENTIDAD + mun_pop.MUN
new_census = sorted(set(mun_pop.key) - set(munis.key))
locs = iter_all[~iter_all.LOC.isin(["0000", "9998", "9999"]) & (iter_all.MUN != "000")].copy()
locs["key"] = locs.ENTIDAD + locs.MUN
locs = locs[locs.key.isin(new_census)]
crosswalk = parent_by_points(locs.LONGITUD.map(dms), locs.LATITUD.map(dms),
                             pd.to_numeric(locs.POBTOT, errors="coerce").fillna(0), locs.key)
new_denue = mini[~mini.key.isin(munis.key) & ~mini.key.isin(crosswalk)]
crosswalk.update(parent_by_points(new_denue.longitud.astype(float), new_denue.latitud.astype(float),
                                  np.ones(len(new_denue)), new_denue.key))
pd.Series(crosswalk, name="parent_2017").rename_axis("key_new").to_csv(DATA / "crosswalk_new_municipalities.csv")
mini["key"] = mini.key.replace(crosswalk)
mun_pop["key"] = mun_pop.key.replace(crosswalk)

mini[["id", "chain", "region", "cve_ent", "cve_mun", "key", "latitud", "longitud", "per_ocu", "fecha_alta"]].to_csv(
    DATA / "minisupers.csv", index=False)

# ── 5. OXXO by municipality and state ────────────────────────────────────────
pop_h = mun_pop.assign(pop=mun_pop.POBTOT.astype(int)).groupby("key").agg(pop=("pop", "sum"), NOM_ENT=("NOM_ENT", "first"))
oxxo_mun = (pop_h.join(mini[mini.chain == "OXXO"].groupby("key").size().rename("stores"))
            .fillna({"stores": 0}).join(munis.set_index("key").NOM_MUN))
oxxo_mun["per10k"] = oxxo_mun.stores / oxxo_mun["pop"] * 1e4
oxxo_mun.reset_index().to_csv(DATA / "oxxo_mun_harmonized.csv", index=False)
oxxo_state = oxxo_mun.groupby("NOM_ENT")[["pop", "stores"]].sum()
oxxo_state["per10k"] = oxxo_state.stores / oxxo_state["pop"] * 1e4
oxxo_state.to_csv(DATA / "oxxo_state.csv")

# ── 6. EMEC retail revenue by state (sector 46, real index, 2018 = 100) ──────
emec_dir = unzip("emec")
emec = pd.read_csv(glob.glob(str(emec_dir / "**/conjunto_de_datos/*indice*.csv"), recursive=True)[0], encoding="utf-8")
emec["MES"] = emec.MES.astype(str).str.strip().astype(int)
emec[emec.CODIGO_ACTIVIDAD == 46].to_csv(DATA / "emec_retail_monthly.csv", index=False)

# ── 7. Simplified boundaries in latitude/longitude for maps ──────────────────
def export(gdf, tolerance, cols, path):
    out = gdf.copy()
    out["geometry"] = out.geometry.simplify(tolerance, preserve_topology=True).make_valid()
    out.to_crs(4326)[cols + ["geometry"]].to_file(path, driver="GeoJSON", COORDINATE_PRECISION=4)

export(states, 1500, ["CVE_ENT", "NOM_ENT"], GEO / "mx_states_2017.geojson")
export(munis, 600, ["key", "CVE_ENT", "NOM_MUN"], GEO / "mx_municipalities_2017.geojson")

print("minisupers", len(mini), "| OXXO", int(oxxo_mun.stores.sum()), "| municipalities", len(oxxo_mun),
      "| population", int(oxxo_mun["pop"].sum()), "| crosswalk", len(crosswalk))
