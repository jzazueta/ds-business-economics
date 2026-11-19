# ch13_pull_uninsured.py -- every passage in each bank's fiscal-2022 10-K that
# mentions uninsured deposits, plus the paragraph before and after it (the number
# is sometimes in a neighbor). Run in Colab (sec.gov is reachable there).
# Output: ch13_data/ch13_uninsured_passages.parquet  (small; commit to data/)
# Safe to re-run: finished banks are saved one file each and skipped next time.

import os, re, time, pathlib, logging
import requests
import pandas as pd
from edgar import Company, set_identity
from edgar.company_reports import TenK

logging.getLogger("edgar").setLevel(logging.ERROR)
IDENTITY = "Jorge Zazueta jorge.zazueta@uaslp.mx"   # <-- check before running
set_identity(IDENTITY)
HEADERS = {"User-Agent": IDENTITY}
OUT = pathlib.Path("ch13_data"); OUT.mkdir(exist_ok=True)
PARTS = OUT / "parts_uninsured_v2"; PARTS.mkdir(exist_ok=True)
# banks describe the figure in several ways, not only with the word "uninsured"
KEYWORD = re.compile(
    r"uninsured"
    r"|(in excess of|exceed(s|ed|ing)?|above|over)\s+(the\s+)?(applicable\s+|standard\s+)?"
    r"(FDIC|federal\s+deposit)?\s*insurance\s+(coverage\s+)?limit"
    r"|not\s+(covered|insured)\s+by\s+(the\s+)?(FDIC|federal\s+deposit\s+insurance)", re.I)

def get(url, tries=3):
    """GET politely (SEC allows 10 requests/second), retrying slow responses."""
    for k in range(tries):
        try:
            time.sleep(0.12)
            r = requests.get(url, headers=HEADERS, timeout=60); r.raise_for_status()
            return r.text
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
                   f"{acc}/{self._f.primary_document}")

corpus = pd.read_parquet("https://raw.githubusercontent.com/jzazueta/ds-business-economics/"
                         "main/data/ch13_filings_10k.parquet",
                         columns=["cik", "name", "accession", "period"])
fy22 = corpus[corpus["period"].str.startswith("2022")]
print(len(fy22), "fiscal-2022 filings")

for i, r in enumerate(fy22.itertuples(), 1):
    part = PARTS / f"{r.cik}.parquet"
    if part.exists():
        continue
    t0, rows = time.time(), []
    try:
        f = [x for x in Company(int(r.cik)).get_filings(form="10-K", amendments=False,
                                                        filing_date="2022-06-01:2023-12-31")
             if x.accession_no == r.accession][0]
        doc = TenK(PrimaryDocFiling(f)).document
        md = doc.to_markdown()
        blocks = [b for b in md.split("\n\n") if b.strip()]      # paragraphs and whole tables
        hits = [k for k, b in enumerate(blocks) if KEYWORD.search(b)]
        keep = sorted({j for k in hits for j in (k - 1, k, k + 1) if 0 <= j < len(blocks)})
        for j in keep:
            rows.append({"cik": r.cik, "accession": r.accession, "block": j,
                         "is_match": j in hits, "passage": blocks[j][:4000]})
        status = f"{len(rows)} passages"
    except Exception as e:
        rows.append({"cik": r.cik, "accession": r.accession, "block": -1,
                     "is_match": False, "passage": f"ERROR: {e!r}"})
        status = "error"
    pd.DataFrame(rows).to_parquet(part, index=False)
    print(f"[{i}/{len(fy22)}] {r.name}: {status}, {time.time()-t0:.0f}s")

out = pd.concat([pd.read_parquet(p) for p in PARTS.glob("*.parquet")], ignore_index=True)
out.to_parquet(OUT / "ch13_uninsured_passages.parquet", index=False)
ok = out[~out["passage"].str.startswith("ERROR")]
print(f"\nbanks with at least one match: {ok['cik'].nunique()} of {len(fy22)}; "
      f"matching blocks: {ok['is_match'].sum()}; with neighbors: {len(ok)}; "
      f"errors: {(out['passage'].str.startswith('ERROR')).sum()}")
print("file size (MB):", round(os.path.getsize(OUT / "ch13_uninsured_passages.parquet") / 1e6, 1))
