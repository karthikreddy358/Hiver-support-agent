import os
from anthropic import Anthropic
from retrieval import ResolutionIndex

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

REPLY_SYSTEM_PROMPT = """You are drafting a customer support reply for {brand} on Twitter/X.
Write in the brand's voice as evidenced by the example resolutions below. Be concise
(fits a tweet reply), empathetic, and concrete about next steps. Do not invent policy
details (refund amounts, timelines) that are not implied by the examples -- if you
don't know, ask the customer to DM order details instead of guessing.

Here is how {brand} has resolved similar issues before:
{examples}
"""


def generate_reply(customer_text: str, brand: str, index: ResolutionIndex,
                    intent: str, k: int = 3, model: str = "claude-sonnet-4-6") -> dict:
    retrieved = index.query(customer_text, k=k)
    examples_str = "\n\n".join(
        f"- Customer: {row.customer_text}\n  {brand}: {row.brand_reply_text}"
        for row in retrieved.itertuples()
    )

    system = REPLY_SYSTEM_PROMPT.format(brand=brand, examples=examples_str)
    user_msg = f"Intent: {intent}\nCustomer message: {customer_text}\n\nDraft the reply."

    resp = client.messages.create(
        model=model,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    return {
        "reply": resp.content[0].text.strip(),
        "grounding_examples": retrieved.to_dict(orient="records"),
    }
