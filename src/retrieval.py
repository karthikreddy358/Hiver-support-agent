"""
Retrieval over historically resolved (customer_text -> brand_reply_text) pairs.
Used to ground reply generation in how the brand actually handled similar issues.

TF-IDF is used deliberately -- it's fast, free, and fully explainable (you can
show *why* an example was retrieved), which matters for the "convince us it's
trustworthy" bar in this assignment. Swapping in embeddings is a natural
"next week" improvement -- see report/REPORT_TEMPLATE.md.
"""
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = Path("data/processed")


class ResolutionIndex:
    def __init__(self, threads_df: pd.DataFrame):
        # only ground on threads that look genuinely resolved (no further complaint)
        self.df = threads_df[threads_df["resolution_signal"] == "no_further_reply"].reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.df["customer_text"])

    def query(self, text: str, k: int = 3) -> pd.DataFrame:
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]
        results = self.df.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results[["customer_text", "brand_reply_text", "similarity"]]

    def save(self, brand: str):
        with open(INDEX_DIR / f"{brand}_index.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(brand: str) -> "ResolutionIndex":
        with open(INDEX_DIR / f"{brand}_index.pkl", "rb") as f:
            return pickle.load(f)
