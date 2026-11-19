# ch13_pull_stage2.py -- second pull for Chapter 13, run in Colab.
#   (1) Item 1A as markdown for every filing, split into individual risk factors
#   (2) balance-sheet facts for every bank from XBRL company facts (repairs deposits)
#
# Setup in Colab:
#   !pip install -U edgartools pyarrow
#   (restart the session)
#   from google.colab import drive; drive.mount('/content/drive')
#   %cd /content/drive/MyDrive
# Put ch13_split_risk_factors.py and ch13_filings_10k.parquet next to this script, then
#   %run ch13_pull_stage2.py
# Safe to re-run: finished banks are saved one file each and skipped next time.

import time, pathlib, logging
import requests
import pandas as pd
from edgar import Company, set_identity
from edgar.company_reports import TenK
from ch13_split_risk_factors import split_risk_factors

logging.getLogger("edgar").setLevel(logging.ERROR)

IDENTITY = "Jorge Zazueta jorge.zazueta@uaslp.mx"   # <-- check before running
set_identity(IDENTITY)
HEADERS = {"User-Agent": IDENTITY}
OUT = pathlib.Path("ch13_data"); OUT.mkdir(exist_ok=True)
PARTS = OUT / "parts_1a"; PARTS.mkdir(exist_ok=True)

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

class PrimaryDocFiling:
    """Let the 10-K parser download only the main document."""
    def __init__(self, filing): self._f = filing
    def __getattr__(self, name): return getattr(self._f, name)
    def html(self):
        acc = self._f.accession_no.replace("-", "")
        return get(f"https://www.sec.gov/Archives/edgar/data/{self._f.cik}/"
                   f"{acc}/{self._f.primary_document}", as_json=False)

def item1a_markdown(filing):
    """Item 1A as markdown, which keeps the bold risk-factor headings."""
    doc = TenK(PrimaryDocFiling(filing)).document
    keys = [k for k in doc.sections.keys() if k.endswith("item_1a")]
    if not keys:
        return None
    md = doc.sections[keys[0]].markdown
    return md() if callable(md) else md

corpus = pd.read_parquet("ch13_filings_10k.parquet", columns=["cik", "name", "accession"])
wanted = corpus.groupby("cik")["accession"].apply(set)

# -- Part 1: Item 1A markdown and risk factors, bank by bank -----------------
for i, (cik, accs) in enumerate(wanted.items(), 1):
    part = PARTS / f"{cik}.parquet"
    if part.exists():
        continue
    t0, rows = time.time(), []
    filings = Company(int(cik)).get_filings(form="10-K", amendments=False,
                                            filing_date="2019-01-01:2024-12-31")
    for f in filings:
        if f.accession_no not in accs:
            continue
        try:
            md = item1a_markdown(f)
            rows.append({"cik": cik, "accession": f.accession_no, "item1a_md": md, "error": None})
        except Exception as e:
            rows.append({"cik": cik, "accession": f.accession_no, "item1a_md": None, "error": repr(e)})
    pd.DataFrame(rows).to_parquet(part, index=False)
    n_rf = [len(split_risk_factors(r["item1a_md"])) if r["item1a_md"] else 0 for r in rows]
    print(f"[{i}/{len(wanted)}] cik {cik}: {len(rows)} filings, risk factors {n_rf}, {time.time()-t0:.0f}s")

md = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
md.to_parquet(OUT / "ch13_item1a_md.parquet", index=False)

rf_rows = []
for r in md.dropna(subset=["item1a_md"]).itertuples():
    for k, f in enumerate(split_risk_factors(r.item1a_md), 1):
        rf_rows.append({"cik": r.cik, "accession": r.accession, "rf_order": k, **f})
rf = pd.DataFrame(rf_rows)
rf.to_parquet(OUT / "ch13_risk_factors.parquet", index=False)

# -- Part 2: balance-sheet facts from XBRL company facts ---------------------
TAGS = ["Assets", "Deposits", "InterestBearingDepositLiabilities",
        "NoninterestBearingDepositLiabilities", "StockholdersEquity",
        "LoansAndLeasesReceivableNetReportedAmount",
        "AvailableForSaleSecuritiesDebtSecurities",
        "HeldToMaturitySecurities", "HeldToMaturitySecuritiesFairValue",
        "DebtSecuritiesHeldToMaturityAmortizedCostAfterAllowanceForCreditLoss",
        "CashAndCashEquivalentsAtCarryingValue"]
facts = []
for cik in wanted.index:
    try:
        cf = get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json")
    except Exception as e:
        print("companyfacts failed:", cik, e); continue
    gaap = cf.get("facts", {}).get("us-gaap", {})
    for tag in TAGS:
        for v in gaap.get(tag, {}).get("units", {}).get("USD", []):
            if v.get("end", "") >= "2018-01-01" and v.get("form") in ("10-K", "10-Q"):
                facts.append({"cik": cik, "tag": tag, "end": v["end"], "val": v["val"],
                              "form": v["form"], "fy": v.get("fy"), "fp": v.get("fp"),
                              "filed": v.get("filed"), "accn": v.get("accn")})
pd.DataFrame(facts).to_csv(OUT / "ch13_balance_sheet_facts.csv", index=False)

# -- Quick report -------------------------------------------------------------
print(f"\nfilings with Item 1A markdown: {md['item1a_md'].notna().sum()} of {len(md)}")
print("errors:", md["error"].notna().sum())
per = rf.groupby("accession").size()
print(f"risk factors: {len(rf)}  per filing: median {per.median():.0f}, "
      f"min {per.min()}, max {per.max()}")
print("filings with fewer than 10 risk factors:", (per < 10).sum() + (len(md) - per.size))
print("balance-sheet facts:", len(facts), " banks:", pd.DataFrame(facts)["cik"].nunique())
