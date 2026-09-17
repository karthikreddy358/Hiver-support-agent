import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from intents import classify_intent
from retrieval import ResolutionIndex
from reply_gen import generate_reply
from escalation import decide_escalation


def run_pipeline(customer_text: str, brand: str, index: ResolutionIndex) -> dict:
    intent_result = classify_intent(customer_text)
    reply_result = generate_reply(
        customer_text, brand, index,
        intent=intent_result["intent"],
    )
    escalation_result = decide_escalation(
        customer_text, intent_result["intent"], intent_result["confidence"],
    )

    return {
        "customer_text": customer_text,
        "intent": intent_result["intent"],
        "intent_confidence": intent_result["confidence"],
        "intent_reason": intent_result["reason"],
        "draft_reply": reply_result["reply"],
        "grounding_examples": reply_result["grounding_examples"],
        "escalate": escalation_result["escalate"],
        "escalation_reason": escalation_result["reason"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    ap.add_argument("--n", type=int, default=50, help="number of messages to run through the pipeline")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    threads = pd.read_csv(f"data/processed/{args.brand}_threads.csv")
    index = ResolutionIndex.load(args.brand)

    sample = threads.sample(n=min(args.n, len(threads)), random_state=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for row in tqdm(sample.itertuples(), total=len(sample)):
            result = run_pipeline(row.customer_text, args.brand, index)
            result["thread_id"] = row.thread_id
            result["reference_reply"] = row.brand_reply_text  # for eval comparison
            f.write(json.dumps(result) + "\n")

    print(f"[pipeline] wrote {len(sample)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
