# Decision log

1. **Brand choice**: Chose `AmazonHelp` because it has a high volume of threads, a diverse set of easily identifiable intents (shipping issues, returns, billing), and clear resolution structures in the dataset.
2. **Intent taxonomy size**: Capped at ~6 core intents (e.g., `delivery_delay`, `refund_or_return`, `billing_dispute`) rather than finer-grained ones, to keep golden-set labeling tractable and per-class sample sizes meaningful.
3. **Retrieval method**: TF-IDF instead of embeddings. This was chosen for explainability, speed, and zero infrastructure cost, at the expense of occasionally missing paraphrased customer queries. Revisit if groundedness scores are low.
4. **"Resolved" definition**: A thread counts as resolved-for-grounding only if the customer didn't reply again. While imperfect (they could have given up), it serves as a robust proxy for a terminating resolution over a 3-turn thread.
5. **Escalation threshold**: Confidence < 0.75 triggers escalation. Chosen as a conservative starting point to prioritize not falsely automating nuanced or confusing requests.
6. **Escalation keyword list**: Hand-picked based on high-risk vectors (e.g., "lawyer", "scam", "fraud"). It is slightly brittle to phrasing variants but guarantees hard stops on obviously legally sensitive threads.
7. **Golden set stratification**: Stratified by intent rather than pure random sampling, to ensure rare intents like `billing_dispute` have enough representation for evaluation.
8. **LLM-judge model choice**: Used `claude-sonnet-4-6` for both generation and judging. Acknowledged the risk of self-preference bias, but prioritized utilizing the same model for consistent capability baselines.
9. **Subsample size**: 3,000 threads per brand instead of full dataset, per assignment instructions — chosen to keep pipeline runtime under 15 minutes for reproduction.
10. **Dropped orphaned threads**: Threads where the parent tweet isn't in the dataset are dropped rather than imputed. This avoids fabricating context or grounding replies on hallucinated priors.
