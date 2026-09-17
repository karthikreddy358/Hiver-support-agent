# Hiver SDE Intern Assignment: AI Support Agent

Hey there! This is my submission for the Hiver SDE Intern assignment. 

I've built an AI support agent that handles customer queries for **AmazonHelp** on Twitter. It's trained on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) dataset. 

Whenever a new customer tweet comes in, this agent does three main things:
1. **Figures out what they want (Intent Classification):** It categorizes the tweet into a few core intents (like tracking a package, requesting a refund, etc.).
2. **Drafts a reply:** Instead of just guessing, it searches through past resolved threads to see how human agents at AmazonHelp handled similar problems, and uses that context to write a helpful reply.
3. **Knows when to step back (Escalation):** If a customer uses high-risk keywords (like "scam" or "lawyer") or the model isn't confident, it flags the ticket to be handed over to a real human.

---

## What's inside the repo?

Here's an overview of how the code is organized:

```text
src/
  data_prep.py      # Downloads and cleans the raw Kaggle dataset, stitching tweets into readable threads.
  build_index.py    # Generates a search index of past resolved tickets.
  intents.py        # Defines the intent categories and connects to Claude for classification.
  retrieval.py      # Uses TF-IDF to find the most relevant past threads.
  reply_gen.py      # Feeds the context to the LLM to generate the final draft.
  escalation.py     # Simple rule-based logic to decide if a human needs to take over.
  pipeline.py       # The main script that runs the whole flow from start to finish.
  baselines.py      # Simple benchmark scripts to compare the AI against.
  eval_harness.py   # Grading script for accuracy and quality checks (LLM-as-a-judge).
eval/
  golden_set.csv    # A hand-labeled testing set of 150-250 rows to evaluate the agent.
report/
  REPORT.md         # My final write-up covering performance, failures, and framing.
decision_log.md     # A quick log of the technical tradeoffs and design decisions I made.
```

---

## How to run it locally (Takes < 15 mins)

You can easily test the pipeline yourself. Assuming you're on a standard Unix bash environment:

```bash
# 1. Setup your virtual environment and install the required packages
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Grab the dataset (You'll need your Kaggle API key configured)
pip install kaggle
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw --unzip

# 3. Add your Anthropic (Claude) API key to the environment
export ANTHROPIC_API_KEY=sk-...

# 4. Filter the data to AmazonHelp and stitch the threads together
python src/data_prep.py --brand AmazonHelp --sample_size 3000

# 5. Build our TF-IDF search index using the historical data
python src/build_index.py --brand AmazonHelp

# 6. Run the main pipeline (testing it on 50 messages)
python src/pipeline.py --brand AmazonHelp --n 50 --out data/processed/predictions.jsonl

# 7. (Optional) Run the evaluation harness
python src/eval_harness.py --golden eval/golden_set.csv --predictions data/processed/predictions.jsonl
```

*(Note for Windows users: Activate your env using `.\venv\Scripts\activate` instead!)*

---

## Why AmazonHelp?

I chose AmazonHelp mostly due to the sheer volume and quality of their data in the dataset. Because they handle everything from misplaced packages to billing bugs, it gave me a really diverse, yet predictable set of intents to map out. (You can read more about this in `decision_log.md`).

## Known Limitations & Future Work

No project is perfect! Here are a few things I'd improve if I had more time:
- **TF-IDF vs Embeddings:** Right now, the retriever uses TF-IDF, which is extremely fast and cheap, but it struggles if the customer paraphrases a lot. Upgrading this to dense embeddings (like `sentence-transformers`) would be my #1 priority.
- **Orphaned Tweets:** The Twitter dataset is slightly messy, meaning some threads are broken where the original parent tweet is missing. I decided to drop these rather than risk hallucinating context.
- **Data Subsampling:** To respect API rate limits and keep the pipeline fast, the system runs on a 3,000 row subsample rather than the entire 3 million row dataset.

## Citations & Borrowed Code

Everything in this repository was largely written from scratch for this assignment, leveraging the provided scaffolding. No external third-party snippets were blindly copy-pasted.
