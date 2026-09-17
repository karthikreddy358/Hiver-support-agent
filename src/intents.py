"""
Intent taxonomy + classifier.

IMPORTANT: The taxonomy below is a STARTER template. Per the assignment, you
must derive your actual intents from reading a sample of your brand's real
customer messages -- don't ship this list unedited. Replace INTENTS after
your own inductive labeling pass (see eval/labeling_guide.md).
"""
import json
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# STARTER taxonomy -- edit based on what you actually see in your brand's data.
INTENTS = {
    "delivery_delay": "Order/shipment is late, lost, or delivery status is unclear.",
    "refund_or_return": "Customer wants a refund, return, or exchange.",
    "account_access": "Login, password, or account lockout issues.",
    "billing_dispute": "Wrong charge, double charge, or unclear billing.",
    "bug_report": "App/website/product not working as expected (technical fault).",
    "product_question": "Pre-purchase or how-to question about a product/feature.",
    "general_complaint": "Dissatisfaction that doesn't fit a more specific bucket.",
    "praise_or_other": "Compliments, small talk, or anything non-actionable.",
}

CLASSIFY_SYSTEM_PROMPT = """You are an intent classifier for a customer support system.
Classify the customer's message into exactly one of the given intents.
Respond ONLY with JSON: {{"intent": "<intent_name>", "confidence": <0-1 float>, "reason": "<one short sentence>"}}
No markdown, no preamble.

Intents:
{intent_list}
"""


def classify_intent(text: str, model: str = "claude-sonnet-4-6") -> dict:
    intent_list = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    system = CLASSIFY_SYSTEM_PROMPT.format(intent_list=intent_list)

    resp = client.messages.create(
        model=model,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": text}],
    )
    raw = resp.content[0].text.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"intent": "general_complaint", "confidence": 0.0, "reason": f"parse_failure: {raw[:100]}"}

    if parsed.get("intent") not in INTENTS:
        parsed["intent"] = "general_complaint"
    return parsed
