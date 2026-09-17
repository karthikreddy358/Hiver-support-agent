"""
Two baselines required by the report:

1. Trivial baseline: always predicts the majority intent, always escalates,
   and replies with a generic canned message. This is your floor.
2. Simple baseline: keyword/rule-based intent classifier (no LLM), nearest-
   neighbor reply (just returns the most similar historical reply verbatim,
   no generation), and the same rule-based escalation logic as the main
   system (so escalation isn't the variable being compared).

Run this and compare its metrics against src/pipeline.py output in
src/eval_harness.py to produce the "Results vs baselines" section of the report.
"""
import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from retrieval import ResolutionIndex
from escalation import decide_escalation

GENERIC_REPLY = "Thanks for reaching out! We're sorry for the trouble -- a member of our team will follow up shortly."

# very rough keyword rules -- deliberately weak, this IS the simple baseline
KEYWORD_RULES = {
    "delivery_delay": ["delivery", "shipped", "shipping", "arrive", "package", "order"],
    "refund_or_return": ["refund", "return", "exchange", "money back"],
    "account_access": ["login", "password", "locked out", "can't sign in", "account access"],
    "billing_dispute": ["charged", "billing", "invoice", "double charge", "overcharged"],
    "bug_report": ["bug", "crash", "not working", "broken", "error"],
    "product_question": ["how do i", "how to", "does it", "can i"],
}


def trivial_predict(customer_text: str) -> dict:
    return {
        "customer_text": customer_text,
        "intent": "general_complaint",
        "intent_confidence": 0.0,
        "draft_reply": GENERIC_REPLY,
        "escalate": True,
        "escalation_reason": "Trivial baseline always escalates.",
    }


def simple_predict(customer_text: str, index: ResolutionIndex) -> dict:
    text_lower = customer_text.lower()
    intent = "general_complaint"
    for candidate_intent, keywords in KEYWORD_RULES.items():
        if any(kw in text_lower for kw in keywords):
            intent = candidate_intent
            break

    retrieved = index.query(customer_text, k=1)
    reply = retrieved.iloc[0]["brand_reply_text"] if len(retrieved) else GENERIC_REPLY

    escalation_result = decide_escalation(customer_text, intent, intent_confidence=0.6)

    return {
        "customer_text": customer_text,
        "intent": intent,
        "intent_confidence": 0.6,  # rule-based, no real confidence signal
        "draft_reply": reply,
        "escalate": escalation_result["escalate"],
        "escalation_reason": escalation_result["reason"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    threads = pd.read_csv(f"data/processed/{args.brand}_threads.csv")
    index = ResolutionIndex.load(args.brand)
    sample = threads.sample(n=min(args.n, len(threads)), random_state=1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for row in tqdm(sample.itertuples(), total=len(sample)):
            trivial = trivial_predict(row.customer_text)
            simple = simple_predict(row.customer_text, index)
            f.write(json.dumps({
                "thread_id": row.thread_id,
                "reference_reply": row.brand_reply_text,
                "trivial": trivial,
                "simple": simple,
            }) + "\n")

    print(f"[baselines] wrote {len(sample)} rows -> {out_path}")


if __name__ == "__main__":
    main()
