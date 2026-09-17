"""
Decides auto-handle vs. escalate-to-human, with a stated reason.

This is a starter rule set -- tune thresholds/keywords against your golden
set failure analysis, and log why you chose them in decision_log.md.
"""

# Intents that should basically never be fully automated (legal/financial/safety risk)
HIGH_RISK_INTENTS = {"billing_dispute", "account_access"}

# Keywords that signal escalation regardless of intent (legal threats, fraud, self-harm, etc.)
ESCALATION_KEYWORDS = [
    "lawyer", "sue", "legal action", "fraud", "scam", "chargeback",
    "unauthorized", "hacked", "never received", "third time",
]

CONFIDENCE_THRESHOLD = 0.75


def decide_escalation(customer_text: str, intent: str, intent_confidence: float) -> dict:
    text_lower = customer_text.lower()
    matched_keywords = [kw for kw in ESCALATION_KEYWORDS if kw in text_lower]

    if matched_keywords:
        return {
            "escalate": True,
            "reason": f"Escalation keyword(s) matched: {matched_keywords}",
        }

    if intent in HIGH_RISK_INTENTS:
        return {
            "escalate": True,
            "reason": f"Intent '{intent}' is high-risk (financial/account access) -- requires human judgment.",
        }

    if intent_confidence < CONFIDENCE_THRESHOLD:
        return {
            "escalate": True,
            "reason": f"Low classifier confidence ({intent_confidence:.2f} < {CONFIDENCE_THRESHOLD}).",
        }

    return {
        "escalate": False,
        "reason": f"Intent '{intent}' is low-risk and classified with confidence {intent_confidence:.2f}.",
    }
