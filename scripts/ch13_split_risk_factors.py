# ch13_split_risk_factors.py -- split a 10-K Item 1A (as markdown) into
# individual risk factors. Each risk factor opens with an emphasized heading
# sentence. Filers emphasize it in different ways, and three are handled:
#   bold paragraph      **Liquidity risk could impair our ability to ...**
#   italic paragraph    *We are subject to interest rate risk.*
#   bold lead-in        **Our allowance may not cover losses.** As a lender, we ...
# Short emphasized lines are category headers ("Credit Risks"), not risk factors.

import re

MIN_HEADING_WORDS = 8          # emphasized lines shorter than this are category headers
NOISE = re.compile(r"^(table of contents|\d{1,3}|page \d+|.{0,60}form 10-k)$", re.I)
EMPH_RUN = re.compile(r"\*{1,3}([^*]+?)\*{1,3}")
LEAD_IN = re.compile(r"^\*\*\*?(?P<head>[^*]+?)\*{2,3}\s*(?P<rest>.*)$", re.S)

def _clean(s):
    """Remove markdown escapes, emphasis and underline markers, collapse whitespace."""
    s = re.sub(r"\\([\\`*_{}\[\]()#+\-.!$&|])", r"\1", s)
    s = re.sub(r"</?u>", "", s).replace("*", "")
    return re.sub(r"\s+", " ", s).strip()

def _fully_emphasized(p):
    """True if the whole paragraph is bold or italic (possibly in several runs)."""
    return p.startswith("*") and _clean(EMPH_RUN.sub("", p)) in ("", ".")

def _classify(p):
    """Return ('heading', head, rest), ('short', text, None) or ('body', text, None)."""
    text = _clean(p)
    if _fully_emphasized(p):
        kind = "heading" if len(text.split()) >= MIN_HEADING_WORDS else "short"
        return kind, text, ""
    m = LEAD_IN.match(p)
    if m:
        head = _clean(m.group("head"))
        if len(head.split()) >= MIN_HEADING_WORDS:
            return "heading", head, _clean(m.group("rest")).lstrip(". ")
    return "body", text, None

def split_risk_factors(md):
    """Return a list of dicts: category, heading, body, n_words, one per risk factor."""
    paras = [p.strip() for p in md.split("\n\n") if p.strip()]
    factors, category, current = [], None, None
    for p in paras:
        if not _clean(p) or NOISE.match(_clean(p)):
            continue                                  # page furniture
        kind, text, rest = _classify(p)
        if kind == "short":
            category = text                           # a category header
        elif kind == "heading":
            current = {"category": category, "heading": text, "body": [rest] if rest else []}
            factors.append(current)
        elif current is not None:
            # a paragraph that starts lowercase continues one broken by a page break
            if current["body"] and text[:1].islower():
                current["body"][-1] += " " + text
            else:
                current["body"].append(text)
    for f in factors:
        f["body"] = "\n\n".join(f["body"])
        f["n_words"] = len((f["heading"] + " " + f["body"]).split())
    return factors
