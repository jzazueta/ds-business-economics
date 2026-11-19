# ch13_build_dataset.py -- build the Chapter 13 analysis tables from the files in
# data/. Needs no SEC access, so anyone with the repo can reproduce the tables.
#
# Inputs (data/):  ch13_banks.csv, ch13_filings_10k.parquet,
#                  ch13_risk_factors.parquet, ch13_balance_sheet_facts.csv
# Outputs (data/): ch13_filings.csv   one row per 10-K, with inclusion flags
#                  ch13_outcomes.csv  one row per bank: Q1-2023 deposit change,
#                                     stress label, end-2022 balance sheet
# Run from the repository root:  python scripts/ch13_build_dataset.py

import pathlib
import numpy as np
import pandas as pd

DATA = pathlib.Path("data")
STRESS_THRESHOLD = -0.05        # a Q1-2023 deposit decline beyond 5% counts as stress
MIN_RISK_FACTORS = 10           # fewer than this means the heading split failed
MIN_ITEM1A_CHARS = 5_000        # shorter than this is a cross-reference, not the section

# Banks whose Q1-2023 deposits changed through an acquisition, not customer flows.
MERGERS_Q1_2023 = {
    887343: "Columbia Banking System completed its merger with Umpqua, March 2023",
    798941: "First Citizens took over Silicon Valley Bridge Bank, March 27, 2023",
    910073: "Flagstar (NYCB) took over Signature Bank deposits, March 20, 2023",
    730708: "Seacoast completed its acquisition of Professional Holding, January 2023",
}
FAILED = {719739: "SVB Financial Group; Silicon Valley Bank failed March 10, 2023"}

banks = pd.read_csv(DATA / "ch13_banks.csv")
filings = pd.read_parquet(DATA / "ch13_filings_10k.parquet")
rf = pd.read_parquet(DATA / "ch13_risk_factors.parquet", columns=["accession"])
facts = pd.read_csv(DATA / "ch13_balance_sheet_facts.csv")

# -- Table 1: one row per filing ----------------------------------------------
f = filings[["cik", "name", "accession", "filing_date", "period"]].copy()
f["fiscal_year"] = f["period"].str[:4].astype(int)
f["item1a_chars"] = filings["item1a"].str.len()
f["n_risk_factors"] = f["accession"].map(rf.groupby("accession").size()).fillna(0).astype(int)

f["in_window"] = f["fiscal_year"].between(2018, 2023)        # drops 3 fiscal-2024 filings
f["item1a_ok"] = f["item1a_chars"] >= MIN_ITEM1A_CHARS       # usable for tone and similarity
f["split_ok"] = f["n_risk_factors"] >= MIN_RISK_FACTORS      # usable for topics and labels
f = f.sort_values(["cik", "fiscal_year"]).reset_index(drop=True)
f.to_csv(DATA / "ch13_filings.csv", index=False)

# -- Table 2: one row per bank ------------------------------------------------
def as_reported(tag, end):
    """Each bank's value for a tag at a date, from the filing that first reported it.
    Later filings can restate the past (after a merger, the accounting acquirer's
    history replaces the target's), so the first report is the one the market saw.
    Only filings made within six months of the date count, so a tag a bank started
    using after a merger cannot bring in restated figures."""
    cutoff = (pd.Timestamp(end) + pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    x = facts[(facts["tag"] == tag) & (facts["end"] == end) & (facts["filed"] <= cutoff)]
    return x.sort_values("filed").groupby("cik")["val"].first()

def first_available(tags, end):
    """Banks label the same line item with different tags. Take the first tag
    in the list that a bank reports; a tuple of tags means add them up."""
    out = pd.Series(dtype=float)
    for t in tags:
        if isinstance(t, tuple):
            v = as_reported(t[0], end)
            for extra in t[1:]:
                v = v.add(as_reported(extra, end))     # NaN unless all parts exist
        else:
            v = as_reported(t, end)
        out = out.combine_first(v)
    return out

# priority lists, built from the tags these 96 banks use at end-2022
TAGS = {
    "deposits": ["Deposits", ("InterestBearingDepositLiabilities",
                              "NoninterestBearingDepositLiabilities")],
    "assets":   ["Assets"],
    "equity":   ["StockholdersEquity",
                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "loans":    ["LoansAndLeasesReceivableNetReportedAmount",
                 "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
                 "FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss"],
    "cash":     ["CashAndCashEquivalentsAtCarryingValue",
                 "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
                 ("CashAndDueFromBanks", "InterestBearingDepositsInBanks")],
    "afs":      ["AvailableForSaleSecuritiesDebtSecurities",
                 "DebtSecuritiesAvailableForSaleExcludingAccruedInterest"],
    "htm":      ["HeldToMaturitySecurities",
                 "DebtSecuritiesHeldToMaturityExcludingAccruedInterestAfterAllowanceForCreditLoss",
                 "DebtSecuritiesHeldToMaturityAmortizedCostAfterAllowanceForCreditLoss"],
    "htm_fv":   ["HeldToMaturitySecuritiesFairValue"],
    # unrealized losses, as tagged in the securities footnote
    "afs_loss": ["AvailableForSaleDebtSecuritiesAccumulatedGrossUnrealizedLossBeforeTax",
                 "DebtSecuritiesAvailableForSaleUnrealizedLossPositionAccumulatedLoss",
                 ("DebtSecuritiesAvailableForSaleContinuousUnrealizedLossPositionLessThan12MonthsAccumulatedLoss",
                  "DebtSecuritiesAvailableForSaleContinuousUnrealizedLossPosition12MonthsOrLongerAccumulatedLoss")],
    "htm_loss": ["HeldToMaturitySecuritiesAccumulatedUnrecognizedHoldingLoss"],
}

END = "2022-12-31"
o = banks[["cik", "name", "sic", "state"]].set_index("cik")
o["deposits_2022q4"] = first_available(TAGS["deposits"], END)
o["deposits_2023q1"] = first_available(TAGS["deposits"], "2023-03-31")
o["deposit_change_q1"] = o["deposits_2023q1"] / o["deposits_2022q4"] - 1
for col in ["assets", "equity", "loans", "cash", "afs", "htm", "htm_fv", "afs_loss", "htm_loss"]:
    o[col] = first_available(TAGS[col], END)

# a bank that reports no HTM book at all holds none; one that reports a book
# but not the loss tag gets the loss from book value minus fair value
o[["afs_loss", "htm_loss"]] = o[["afs_loss", "htm_loss"]].abs()   # some banks tag losses as negatives
no_htm = o[["htm", "htm_fv", "htm_loss"]].isna().all(axis=1)
o.loc[no_htm, ["htm", "htm_loss"]] = 0.0
o["htm_loss"] = o["htm_loss"].fillna((o["htm"] - o["htm_fv"]).clip(lower=0))
o["htm"] = o["htm"].fillna(o["htm_fv"] + o["htm_loss"])      # book value = fair value + loss
o["afs"] = o["afs"].fillna(0.0)
o["afs_loss"] = o["afs_loss"].where(o["afs"] > 0, 0.0)

# ratios a credit analyst would read first
o["log_assets"] = np.log(o["assets"])
o["equity_to_assets"] = o["equity"] / o["assets"]
o["loans_to_assets"] = o["loans"] / o["assets"]
o["cash_to_assets"] = o["cash"] / o["assets"]
o["securities_to_assets"] = (o["afs"] + o["htm"]) / o["assets"]
o["unrealized_loss_to_equity"] = (o["afs_loss"] + o["htm_loss"]) / o["equity"]
o["htm_loss_to_equity"] = o["htm_loss"] / o["equity"]

o["merger_q1_2023"] = o.index.isin(MERGERS_Q1_2023.keys())
o["failed"] = o.index.isin(FAILED.keys())
o["stress"] = np.where(o["failed"], True,
              np.where(o["deposit_change_q1"].isna() | o["merger_q1_2023"], np.nan,
                       o["deposit_change_q1"] < STRESS_THRESHOLD))
o.reset_index().to_csv(DATA / "ch13_outcomes.csv", index=False)

# -- Report -------------------------------------------------------------------
print(f"filings: {len(f)}  in window: {f['in_window'].sum()}  "
      f"Item 1A usable: {(f['in_window'] & f['item1a_ok']).sum()}  "
      f"split usable: {(f['in_window'] & f['split_ok']).sum()}")
lab = o["stress"].dropna().astype(bool)
print(f"banks: {len(o)}  labeled: {len(lab)}  stressed: {lab.sum()}  "
      f"(failed {o['failed'].sum()}, merger-excluded {o['merger_q1_2023'].sum()}, "
      f"no deposit data {o['deposit_change_q1'].isna().sum() - o['failed'].sum()})")
print(o.loc[o["stress"] == True, ["name", "deposit_change_q1", "unrealized_loss_to_equity"]]
      .sort_values("deposit_change_q1").round(3).to_string())
