"""
Loads the raw Kaggle 'Customer Support on Twitter' twcs.csv, filters to a single
brand, reconstructs (customer_message -> brand_reply) threads, and writes a
subsample to data/processed/.

Usage:
    python src/data_prep.py --brand AmazonHelp --sample_size 3000
"""
import argparse
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/twcs.csv")
OUT_DIR = Path("data/processed")


def load_raw() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_PATH} not found. Download via:\n"
            "  kaggle datasets download -d thoughtvector/customer-support-on-twitter "
            "-p data/raw --unzip"
        )
    df = pd.read_csv(RAW_PATH)
    # expected columns: tweet_id, author_id, inbound, created_at, text,
    # response_tweet_id, in_response_to_tweet_id
    return df


def reconstruct_threads(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """
    Build (customer_tweet -> brand_reply) pairs for a given brand handle.
    A brand reply is an outbound tweet authored by `brand` whose
    in_response_to_tweet_id points at an inbound (customer) tweet.
    """
    df = df.copy()
    df["text"] = df["text"].astype(str)

    brand_replies = df[(df["author_id"] == brand) & (~df["inbound"])]
    customer_tweets = df[df["inbound"]].set_index("tweet_id")

    rows = []
    dropped = 0
    for _, reply in brand_replies.iterrows():
        parent_id = reply.get("in_response_to_tweet_id")
        if pd.isna(parent_id) or parent_id not in customer_tweets.index:
            dropped += 1
            continue
        customer_msg = customer_tweets.loc[parent_id]

        # walk forward one more step to see if the customer replied again
        # (signals whether the brand's reply actually resolved the issue)
        follow_up = df[df["in_response_to_tweet_id"] == reply["tweet_id"]]
        resolved_signal = "no_further_reply" if follow_up.empty else "customer_replied_again"

        rows.append({
            "thread_id": f"{parent_id}_{reply['tweet_id']}",
            "customer_tweet_id": parent_id,
            "customer_text": customer_msg["text"],
            "customer_created_at": customer_msg["created_at"],
            "brand_reply_id": reply["tweet_id"],
            "brand_reply_text": reply["text"],
            "brand_reply_created_at": reply["created_at"],
            "resolution_signal": resolved_signal,
        })

    print(f"[data_prep] {brand}: {len(rows)} threads reconstructed, {dropped} dropped (orphaned parent).")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True, help="Brand author_id, e.g. AmazonHelp")
    ap.add_argument("--sample_size", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    threads = reconstruct_threads(df, args.brand)

    if len(threads) > args.sample_size:
        threads = threads.sample(n=args.sample_size, random_state=args.seed)

    out_path = OUT_DIR / f"{args.brand}_threads.csv"
    threads.to_csv(out_path, index=False)
    print(f"[data_prep] wrote {len(threads)} rows -> {out_path}")


if __name__ == "__main__":
    main()
