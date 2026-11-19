# ch13_embed_risk_factors.py -- sentence embeddings for every risk factor used in
# Section 13.5 (topics). Run once in Colab; commit the output to data/.
#   Runtime > Change runtime type > T4 GPU  (about 1-2 minutes; CPU works, ~10-15 min)
#   (sentence-transformers comes preinstalled in Colab; if not: !pip install -q sentence-transformers)
#   %run ch13_embed_risk_factors.py
# Output: ch13_data/ch13_rf_embeddings.parquet  (~30 MB; one row per risk factor)

import os, pathlib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE = "https://raw.githubusercontent.com/jzazueta/ds-business-economics/main/data/"
OUT = pathlib.Path("ch13_data"); OUT.mkdir(exist_ok=True)

rf = pd.read_parquet(BASE + "ch13_risk_factors.parquet")
filings = pd.read_csv(BASE + "ch13_filings.csv")
usable = filings.loc[filings["in_window"] & filings["split_ok"], "accession"]
rf = rf[rf["accession"].isin(usable)].reset_index(drop=True)
print(len(rf), "risk factors")                       # expect 18006

# heading first: it states the risk in one sentence. The model reads at most
# 256 word pieces (about 190 words), so long bodies are cut after their opening.
texts = (rf["heading"].str.strip() + ". " + rf["body"].str.strip()).tolist()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL, device=device)
print("device:", device, " max tokens:", model.max_seq_length)
emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                   normalize_embeddings=True, convert_to_numpy=True)

out = rf[["cik", "accession", "rf_order"]].copy()
out["embedding"] = list(emb.astype(np.float32))
out.to_parquet(OUT / "ch13_rf_embeddings.parquet", index=False)

print("shape:", emb.shape, " first values:", np.round(emb[0, :4], 4))
print("file size (MB):", round(os.path.getsize(OUT / "ch13_rf_embeddings.parquet") / 1e6, 1))
