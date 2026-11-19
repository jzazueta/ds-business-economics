# ch13_pull_edgar.py -- build the Chapter 13 corpus from SEC EDGAR.
# Run in Colab (sec.gov is reachable there):  !pip install -U edgartools pyarrow
# then restart the session and:               %run ch13_pull_edgar.py
# Safe to re-run: finished banks are saved one file each and skipped next time.
# Outputs go to ./ch13_data/ ; commit them to data/ch13/ in the manuscript repo.

import time, pathlib, logging
import requests
import pandas as pd
from edgar import Company, set_identity
from edgar.company_reports import TenK

logging.getLogger("edgar").setLevel(logging.ERROR)   # silence parser warnings

# SEC requires a real name and email in every request's User-Agent.
IDENTITY = "Jorge Zazueta jorge.zazueta@uaslp.mx"   # <-- check before running
set_identity(IDENTITY)
HEADERS = {"User-Agent": IDENTITY}
OUT = pathlib.Path("ch13_data"); OUT.mkdir(exist_ok=True)
PARTS = OUT / "parts"; PARTS.mkdir(exist_ok=True)     # one file per bank

BANK_SIC = {"6021", "6022", "6035", "6036"}   # commercial banks and savings institutions
MIN_ASSETS, MAX_ASSETS = 10e9, 250e9          # "regional": above $10B, below the megabanks
FILED_FROM, FILED_TO = "2019-01-01", "2024-12-31"   # 10-Ks for fiscal years 2018-2023

def get(url, as_json=True, tries=3):
    """GET politely (SEC allows 10 requests/second), retrying slow responses."""
    for k in range(tries):
        try:
            time.sleep(0.12)
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r.json() if as_json else r.text
        except Exception:
            if k == tries - 1: raise
            time.sleep(5 * (k + 1))

def frame(tag, period):
    """One XBRL 'frame': a single tag, for every filer, at one period end."""
    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/USD/{period}.json"
    return pd.DataFrame(get(url)["data"])[["cik", "entityName", "end", "val", "accn"]]

class PrimaryDocFiling:
    """Wrap an EDGAR filing so the 10-K parser downloads only the main document,
    not the full submission with every exhibit and XBRL file."""
    def __init__(self, filing):
        self._f = filing
    def __getattr__(self, name):                   # everything else: the real filing
        return getattr(self._f, name)
    def html(self):
        acc = self._f.accession_no.replace("-", "")
        url = (f"https://www.sec.gov/Archives/edgar/data/"
               f"{self._f.cik}/{acc}/{self._f.primary_document}")
        return get(url, as_json=False)

# -- Step 1: candidate filers by size, then keep the banks -------------------
if (OUT / "banks.csv").exists():
    banks = pd.read_csv(OUT / "banks.csv")
else:
    assets = frame("Assets", "CY2022Q4I")          # total assets at end-2022
    cand = assets[assets["val"].between(MIN_ASSETS, MAX_ASSETS)]
    print(f"{len(cand)} filers with assets in range; checking SIC codes...")
    rows = []
    for cik in cand["cik"]:
        try:
            sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
        except Exception as e:
            print("submissions failed:", cik, e); continue
        if sub.get("sic") in BANK_SIC:
            rows.append({"cik": cik, "name": sub["name"], "sic": sub["sic"],
                         "sic_desc": sub.get("sicDescription"),
                         "state": (sub.get("addresses", {}).get("business", {}) or {}).get("stateOrCountry")})
    banks = pd.DataFrame(rows).merge(
        assets[["cik", "val"]].rename(columns={"val": "assets_2022q4"}), on="cik")
    banks.to_csv(OUT / "banks.csv", index=False)
print(f"{len(banks)} banks")

# -- Step 2: deposits at end-2022 and end-Q1-2023 (the outcome) --------------
if not (OUT / "deposits_frames.csv").exists():
    dep = []
    for period in ["CY2022Q4I", "CY2023Q1I"]:
        d = frame("Deposits", period)
        dep.append(d[d["cik"].isin(banks["cik"])].assign(period=period))
    pd.concat(dep).to_csv(OUT / "deposits_frames.csv", index=False)
    # SVB Financial Group never filed a Q1-2023 10-Q, so it has no Q1 value.

# -- Step 3: 10-K text, Item 1 and Item 1A, bank by bank ---------------------
failures = []
for i, b in enumerate(banks.itertuples(), 1):
    part = PARTS / f"{b.cik}.parquet"
    if part.exists():
        continue                                   # done in an earlier run
    t0, records = time.time(), []
    try:
        filings = Company(int(b.cik)).get_filings(
            form="10-K", amendments=False, filing_date=f"{FILED_FROM}:{FILED_TO}")
    except Exception as e:
        failures.append({"cik": b.cik, "accession": None, "error": repr(e)}); continue
    for f in filings:
        try:
            tenk = TenK(PrimaryDocFiling(f))       # parse the main document only
            records.append({
                "cik": b.cik, "name": b.name, "accession": f.accession_no,
                "filing_date": str(f.filing_date), "period": str(f.report_date),
                "item1": tenk.business, "item1a": tenk.risk_factors})
        except Exception as e:
            failures.append({"cik": b.cik, "accession": f.accession_no, "error": repr(e)})
    pd.DataFrame(records).to_parquet(part, index=False)
    print(f"[{i}/{len(banks)}] {b.name}: {len(records)} filings, {time.time()-t0:.0f}s")

corpus = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
corpus.to_parquet(OUT / "filings_10k.parquet", index=False)
pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)

# -- Step 4: one raw filing, to show readers what the parser hides ------------
svb = banks[banks["name"].str.contains("SVB", case=False)]
if len(svb) and not (OUT / "svb_10k_fy2022_raw.html").exists():
    f = Company(int(svb["cik"].iloc[0])).get_filings(
        form="10-K", amendments=False, filing_date="2023-01-01:2023-12-31")[0]
    (OUT / "svb_10k_fy2022_raw.html").write_text(PrimaryDocFiling(f).html())

# -- Quick report -------------------------------------------------------------
print(f"filings: {len(corpus)}  failures this run: {len(failures)}")
print("missing Item 1A:", corpus["item1a"].isna().sum(),
      " missing Item 1:", corpus["item1"].isna().sum())
print(corpus.groupby(corpus["period"].str[:4]).size())
