# ch13_llm_label_v2.py -- second labeling run, with the "main subject" rule (v2).
# Is the main subject of each risk factor the risk that depositors withdraw their
# deposits? Run in Colab. Output goes to data/. The first run (ch13_llm_label.py,
# rule v1: "one clear sentence is enough") is kept as ch13_llm_labels.parquet.
#
# Setup (once):
#   1. Get an API key at https://platform.claude.com (billing is separate from a
#      Claude.ai subscription). In Colab: key icon on the left > Add new secret >
#      name ANTHROPIC_API_KEY, paste the key, turn on "Notebook access".
#   2. !pip install -q anthropic
#   3. %run ch13_llm_label_v2.py
#
# What it labels: every fiscal-2022 and fiscal-2023 risk factor that mentions
# "deposit" (1,760) with Haiku; the 100 validation items again with Haiku (to
# measure repeatability) and with Sonnet (to compare models).
# Cost: a few dollars at most (the first run used ~1.4k input tokens per call).
# Safe to re-run: finished labels are saved in ch13_data/parts_llm/ and skipped.
#
# Output: ch13_data/ch13_llm_labels_v2.parquet
#   cik, accession, rf_order, model, run, rule, outflow_risk, evidence,
#   input_tokens, output_tokens

import os, json, time, pathlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import anthropic

try:                                      # Colab secret, if running in Colab
    from google.colab import userdata
    os.environ["ANTHROPIC_API_KEY"] = userdata.get("ANTHROPIC_API_KEY")
except ImportError:
    pass                                  # elsewhere: set the environment variable

BASE = "https://raw.githubusercontent.com/jzazueta/ds-business-economics/main/data/"
OUT = pathlib.Path("ch13_data"); OUT.mkdir(exist_ok=True)
PARTS = OUT / "parts_llm_v2"; PARTS.mkdir(exist_ok=True)
MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5"]

SYSTEM = """You read risk factors from the 10-K annual reports of US banks.
Decide whether the MAIN SUBJECT of the risk factor is the bank's liquidity or
deposit funding AND it discusses the possibility that deposits decline or that
depositors withdraw them.

The main subject is what the heading states and most of the text is about. A risk
factor about interest rates, competition, the economy, acquisitions, technology,
natural disasters, goodwill or any other subject gets false, even if it lists a
decline in deposits among the possible consequences.

Answer true for: liquidity or funding risk factors that describe deposit
withdrawals, outflows, runs or the loss of deposits; risk factors about reliance
on deposits that may decrease or be hard to retain; risk factors about uninsured
or large depositors who could leave; risk factors about events in the banking
industry, such as the 2023 bank failures, that may lead depositors to withdraw.

Answer false for: risk factors mainly about something else that mention deposits
in passing; deposit insurance premiums; the cost of deposits when nothing is said
about deposits leaving; deposit growth.

In "evidence", copy word for word the one sentence that best supports a true
answer; if the answer is false, return an empty string."""

SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string"},
        "outflow_risk": {"type": "boolean"},
    },
    "required": ["evidence", "outflow_risk"],
    "additionalProperties": False,
}

client = anthropic.Anthropic(max_retries=6)

def label(heading, body, model):
    """One risk factor in, one label and one evidence sentence out."""
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Heading: {heading}\n\n{body}"}],
        output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
    )
    text = next(b.text for b in msg.content if b.type == "text")
    out = json.loads(text)
    out.update(input_tokens=msg.usage.input_tokens,
               output_tokens=msg.usage.output_tokens)
    return out

# -- the risk factors to label --------------------------------------------------
filings = pd.read_csv(BASE + "ch13_filings.csv")
ok = filings[filings["in_window"] & filings["item1a_ok"] & filings["split_ok"]]
rf = pd.read_parquet(BASE + "ch13_risk_factors.parquet")
rf = rf.merge(ok[["accession", "fiscal_year"]], on="accession")
text = rf["heading"] + " " + rf["body"]
cand = rf[rf["fiscal_year"].isin([2022, 2023])
          & text.str.contains(r"\bdeposit", case=False)].reset_index(drop=True)
print(len(cand), "risk factors to label")                   # expect 1760

val = pd.read_csv(BASE + "ch13_validation_keys.csv")
keys = ["cik", "accession", "rf_order"]
in_val = cand.merge(val[keys], on=keys, how="inner")
print(len(in_val), "of them in the validation set")         # expect 100

jobs = [("claude-haiku-5-5", 1, r) for r in cand.itertuples()]
jobs += [("claude-haiku-5-5", 2, r) for r in in_val.itertuples()]
jobs += [("claude-sonnet-5-5", 1, r) for r in in_val.itertuples()]

# -- run, saving each label as it arrives ---------------------------------------
lock = threading.Lock()
def run(model, run_no, r):
    part = PARTS / f"{model}_{run_no}_{r.cik}_{r.accession}_{r.rf_order}.json"
    if part.exists():
        return "skip"
    out = label(r.heading, r.body, model)
    out.update(cik=int(r.cik), accession=r.accession, rf_order=int(r.rf_order),
               model=model, run=run_no, rule="v2")
    with lock:
        part.write_text(json.dumps(out))
    return "done"

t0, done = time.time(), 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [pool.submit(run, *j) for j in jobs]
    for f in as_completed(futures):
        try:
            f.result(); done += 1
        except Exception as e:
            print("failed:", repr(e)[:200])
        if done % 200 == 0:
            print(f"{done}/{len(jobs)}  {time.time() - t0:.0f}s")

labels = pd.DataFrame([json.loads(p.read_text()) for p in PARTS.glob("*.json")])
labels.to_parquet(OUT / "ch13_llm_labels_v2.parquet", index=False)
print("\nlabels:", len(labels), "of", len(jobs))
print(labels.groupby(["model", "run"])["outflow_risk"].agg(["size", "mean"]).round(3))
print(labels.groupby("model")[["input_tokens", "output_tokens"]].sum())
