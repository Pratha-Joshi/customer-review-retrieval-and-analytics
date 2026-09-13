"""Schema-inspecting, loss-conscious review preprocessing."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

REQUIRED = {"ProductId", "Score", "Text"}

def load_reviews(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Keep source columns; drop only records unusable as review documents."""
    out = df.copy()
    out["Text"] = out["Text"].fillna("").astype(str).str.strip()
    out = out[out["Text"].ne("") & out["ProductId"].notna() & out["Score"].notna()].copy()
    out["Summary"] = out.get("Summary", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    # Defaults must be index-aligned Series; a scalar default cannot be cleaned with Series methods.
    zeroes = pd.Series(0, index=out.index)
    out["HelpfulnessNumerator"] = pd.to_numeric(out.get("HelpfulnessNumerator", zeroes), errors="coerce").fillna(0).clip(lower=0)
    out["HelpfulnessDenominator"] = pd.to_numeric(out.get("HelpfulnessDenominator", zeroes), errors="coerce").fillna(0).clip(lower=0)
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out = out[out["Score"].between(1, 5)].copy()
    den = out["HelpfulnessDenominator"]
    out["helpfulness_ratio"] = (out["HelpfulnessNumerator"] / den.where(den > 0)).fillna(0.0)
    out["review_length"] = out["Text"].str.len()
    out["word_count"] = out["Text"].str.split().str.len()
    time = pd.to_datetime(out.get("Time", pd.Series(pd.NaT, index=out.index)), unit="s", errors="coerce")
    out["review_time"] = time
    out["review_year"] = time.dt.year
    out["rating_bucket"] = pd.cut(out["Score"], [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
    return out.reset_index(drop=True)

def make_documents(df: pd.DataFrame) -> list[dict]:
    docs = []
    # itertuples is materially faster than iterrows on the 500K+ record corpus.
    for pos, row in enumerate(df.itertuples(index=False)):
        source_id = getattr(row, "Id", pos)
        product_id, review_text, summary = row.ProductId, row.Text, row.Summary
        stable_id = "review_" + hashlib.sha1(f"{source_id}|{product_id}|{review_text}".encode()).hexdigest()[:16]
        text = (f"{summary}\n{review_text}").strip()
        review_time = row.review_time
        metadata = {"review_id": str(source_id), "product_id": str(product_id), "rating": float(row.Score),
                    "review_title": summary, "review_time": review_time.isoformat() if pd.notna(review_time) else None,
                    "helpfulness_num": int(row.HelpfulnessNumerator), "helpfulness_den": int(row.HelpfulnessDenominator),
                    "helpfulness_ratio": float(row.helpfulness_ratio), "word_count": int(row.word_count),
                    "rating_bucket": str(row.rating_bucket)}
        docs.append({"id": stable_id, "text": text, "metadata": metadata})
    return docs

def write_jsonl(documents: list[dict], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in documents: f.write(json.dumps(doc, ensure_ascii=False) + "\n")

def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f: return [json.loads(line) for line in f if line.strip()]

def eda_summary(df: pd.DataFrame) -> dict:
    return {"review_count": len(df), "product_count": int(df.ProductId.nunique()), "rating_distribution": df.Score.value_counts().sort_index().to_dict(),
            "missing_values": df.isna().sum().to_dict(), "review_length": df.review_length.describe().to_dict(),
            "reviews_per_product": df.groupby("ProductId").size().describe().to_dict(), "helpfulness": df.helpfulness_ratio.describe().to_dict()}
