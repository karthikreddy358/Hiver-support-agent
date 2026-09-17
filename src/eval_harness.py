"""
Evaluation harness.

Two things happen here:
1. Automated metrics against your hand-labeled golden set (intent accuracy/F1,
   escalation precision/recall against your own judgment of what SHOULD escalate).
2. LLM-as-judge scoring of reply quality (groundedness, correctness, tone),
   which you must then validate against a human sample -- see
   `--judge_agreement_sample` below. Report agreement %, not just judge scores;
   an unvalidated judge is not evidence.

Usage:
    python src/eval_harness.py --golden eval/golden_set.csv --predictions data/processed/predictions.jsonl
"""
import argparse
import json
import os
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

JUDGE_SYSTEM_PROMPT = """You are grading a customer support reply for quality. Score 1-5 on:
- groundedness: is it consistent with the example resolutions it was given, without inventing policy?
- correctness: does it actually address the customer's issue?
- tone: is it appropriately empathetic and professional?

Respond ONLY with JSON: {"groundedness": <1-5>, "correctness": <1-5>, "tone": <1-5>, "notes": "<one sentence>"}
"""


def load_predictions(path: str) -> dict:
    preds = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            preds[row["thread_id"]] = row
    return preds


def intent_metrics(golden: pd.DataFrame, preds: dict) -> None:
    y_true, y_pred = [], []
    for _, row in golden.iterrows():
        pred = preds.get(row["thread_id"])
        if pred is None:
            continue
        y_true.append(row["gold_intent"])
        y_pred.append(pred["intent"])

    print("\n=== Intent classification ===")
    print(classification_report(y_true, y_pred, zero_division=0))


def escalation_metrics(golden: pd.DataFrame, preds: dict) -> None:
    y_true, y_pred = [], []
    for _, row in golden.iterrows():
        pred = preds.get(row["thread_id"])
        if pred is None:
            continue
        y_true.append(bool(row["gold_should_escalate"]))
        y_pred.append(bool(pred["escalate"]))

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print("\n=== Escalation decision ===")
    print(f"Precision: {p:.2f}  Recall: {r:.2f}  F1: {f1:.2f}")
    print("NOTE: for support triage, recall on 'should escalate' usually matters more than "
          "precision -- a missed escalation is worse than an unnecessary one. State this "
          "tradeoff explicitly in your report.")


def llm_judge_reply(customer_text: str, draft_reply: str, grounding_examples: list,
                     model: str = "claude-sonnet-4-6") -> dict:
    examples_str = "\n".join(f"- {e['customer_text']} -> {e['brand_reply_text']}" for e in grounding_examples)
    user_msg = (
        f"Customer message: {customer_text}\n\n"
        f"Grounding examples given to the model:\n{examples_str}\n\n"
        f"Draft reply to grade: {draft_reply}"
    )
    resp = client.messages.create(
        model=model, max_tokens=200,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    try:
        return json.loads(resp.content[0].text.strip())
    except json.JSONDecodeError:
        return {"groundedness": None, "correctness": None, "tone": None, "notes": "parse_failure"}


def run_llm_judge(preds: dict, out_path: str) -> pd.DataFrame:
    rows = []
    for thread_id, pred in preds.items():
        scores = llm_judge_reply(pred["customer_text"], pred["draft_reply"], pred.get("grounding_examples", []))
        scores["thread_id"] = thread_id
        rows.append(scores)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\n=== LLM-judge reply quality (mean scores) ===")
    print(df[["groundedness", "correctness", "tone"]].mean(numeric_only=True))
    print(
        "\nIMPORTANT: these judge scores are NOT evidence on their own. Hand-score a "
        "~30-50 row sample yourself and compute judge-human agreement "
        "(e.g. % within 1 point, or Cohen's kappa on a binned pass/fail) before "
        "citing this number in your report."
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True, help="hand-labeled golden set CSV")
    ap.add_argument("--predictions", required=True, help="pipeline output JSONL")
    ap.add_argument("--judge_out", default="data/processed/judge_scores.csv")
    args = ap.parse_args()

    golden = pd.read_csv(args.golden)
    preds = load_predictions(args.predictions)

    intent_metrics(golden, preds)
    escalation_metrics(golden, preds)
    run_llm_judge(preds, args.judge_out)


if __name__ == "__main__":
    main()
