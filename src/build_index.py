import argparse
import pandas as pd
from pathlib import Path
from retrieval import ResolutionIndex

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    args = ap.parse_args()

    threads_path = Path(f"data/processed/{args.brand}_threads.csv")
    df = pd.read_csv(threads_path)
    idx = ResolutionIndex(df)
    idx.save(args.brand)
    print(f"[build_index] indexed {len(idx.df)} resolved threads for {args.brand}")
