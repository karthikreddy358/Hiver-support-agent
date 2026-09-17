# Hiver SDE Intern Assignment — AI Support Agent

An AI support agent for a single brand from the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Given an incoming customer tweet, the agent:

1. **Classifies** it into one of a small set of intents derived from the data.
2. **Drafts a reply**, grounded in how the brand historically resolved similar issues (retrieval over past resolved threads).
3. **Decides** auto-handle vs. escalate-to-human, with a stated reason.

## Repo structure

```
src/
  data_prep.py      # loads twcs.csv, filters to one brand, reconstructs threads
  build_index.py    # builds the retrieval index of past resolved (customer -> brand reply) pairs
  intents.py         # intent taxonomy + LLM-based classifier
  retrieval.py        # TF-IDF retrieval over historical resolutions
  reply_gen.py        # grounded reply generation
  escalation.py       # auto-handle vs escalate logic
  pipeline.py          # wires the above end to end
  baselines.py         # trivial + simple baselines
  eval_harness.py      # automated metrics + LLM-as-judge, judge-vs-human agreement
eval/
  golden_set_template.csv   # fill this in by hand (150-250 rows)
  labeling_guide.md
report/
  REPORT_TEMPLATE.md
decision_log.md
requirements.txt
```

## Setup (reproduce in < 15 min)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Get the data (requires a free Kaggle account)
pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip

# 2. Set your LLM API key
export ANTHROPIC_API_KEY=sk-...

# 3. Filter to one brand + build a subsample + reconstruct threads
python src/data_prep.py --brand AmazonHelp --sample_size 3000

# 4. Build the retrieval index over historically resolved threads
python src/build_index.py --brand AmazonHelp

# 5. Run the pipeline on a batch of incoming messages
python src/pipeline.py --brand AmazonHelp --n 50 --out data/processed/predictions.jsonl

# 6. Run baselines for comparison
python src/baselines.py --brand AmazonHelp --n 50 --out data/processed/baseline_predictions.jsonl

# 7. Hand-label eval/golden_set_template.csv (150-250 rows), then run the harness
python src/eval_harness.py --golden eval/golden_set.csv --predictions data/processed/predictions.jsonl
```

## Why this brand / these intents

Fill in once you've picked a brand — see `decision_log.md`.

## Known limitations

- Retrieval grounding is TF-IDF, not embeddings — fast/cheap/explainable but misses paraphrases. Noted as a "next week" improvement in the report.
- Thread reconstruction assumes the dataset's `response_tweet_id` / `in_response_to_tweet_id` chain is clean; a fraction of threads are broken/orphaned and are dropped (see `data_prep.py` logs).
- We do not run on the full 3M-row dataset — a per-brand subsample is used throughout, per the assignment's own instructions.

## Citations / borrowed code

Log anything you borrow here (Stack Overflow snippets, tutorial code, etc.) with links. Currently: none — scaffold is original.
