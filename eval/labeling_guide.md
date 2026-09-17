# Labeling guide for the golden set

## Sampling (write this up in the report too)

- Target 150-250 rows.
- **Stratify by intent**: don't just take a random sample — you'll get 80% "general_complaint"
  and no signal on rare-but-important intents like `billing_dispute`. Pull roughly equal
  buckets per intent (run your classifier first, unlabeled, to get a rough bucket count,
  then sample within buckets).
- Include some deliberately hard/ambiguous cases (multi-intent messages, sarcasm, very short
  tweets) — these are where you'll find real failure modes.
- Note the exact sampling procedure (seed, strata sizes) in `decision_log.md` so it's reproducible.

## Labeling each row

For each row, a human (you) fills in:
- `gold_intent`: the intent you'd assign, from your finalized taxonomy in `src/intents.py`.
- `gold_should_escalate`: would a competent human agent escalate this, yes/no? Use your own
  judgment of risk (financial, legal, safety, repeat-complaint) — not the model's output.
- `gold_escalation_reason`: one sentence why.
- `gold_reply_notes`: optional — anything a good reply *must* contain (e.g. "must ask for order number"),
  used later for spot-checking the LLM judge, not for automated scoring.

## Tips

- Label blind to the model's predictions where possible (label from `data/processed/{brand}_threads.csv`
  directly, not from `predictions.jsonl`), so you're not anchored by the model's answer.
- Keep a "disagreed with myself" pass: re-label ~10% of rows a day later and check consistency —
  this becomes a nice footnote in "what's misleading about my headline number."
