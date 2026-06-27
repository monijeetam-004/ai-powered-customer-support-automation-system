"""
llm.py
Lightweight LLM wrapper used by the classification, agent, and supervisor
nodes.

Design choice: this project must be demonstrable without requiring anyone to
provide a paid API key. So `llm_complete()` first checks for an
ANTHROPIC_API_KEY environment variable:
  - If present, it calls the real Claude API (via the `anthropic` package).
  - If absent, it falls back to a deterministic rule-based responder so the
    whole graph (classification, agents, supervisor) still runs end-to-end
    for grading/demo purposes.

To run with a real LLM: `export ANTHROPIC_API_KEY=sk-ant-...` before
launching main.py. No code changes are required.
"""

import os
import re

USE_REAL_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

if USE_REAL_LLM:
    import anthropic
    _client = anthropic.Anthropic()


def llm_complete(system_prompt: str, user_prompt: str, fallback: str) -> str:
    """Call the real LLM if a key is configured, else return `fallback`
    (the fallback is produced by deterministic rule-based logic in nodes.py,
    so the system remains fully functional offline)."""
    if not USE_REAL_LLM:
        return fallback
    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()
    except Exception as e:
        return fallback + f"\n[LLM call failed, used fallback: {e}]"


# ---------------------------------------------------------------------------
# Rule-based intent classifier (used as the deterministic fallback / offline
# brain for Task 3). Mirrors what an LLM-based classifier prompt would do.
# ---------------------------------------------------------------------------

INTENT_KEYWORDS = {
    "Memory": [
        r"previous (issue|problem|request|query)",
        r"what (did|was) (i|my)",
        r"earlier (issue|conversation|problem)",
        r"last time",
        r"history",
    ],
    "Sales": [
        r"pricing", r"price", r"plan", r"subscription plan", r"cost", r"upgrade",
        r"trial", r"discount", r"buy", r"purchase",
    ],
    "Technical": [
        r"crash", r"error", r"bug", r"install", r"installation", r"login",
        r"log in", r"configuration", r"not working", r"freeze", r"upload fail",
    ],
    "Billing": [
        r"refund", r"invoice", r"bill", r"payment", r"charge", r"overcharge",
        r"cancel.*subscription", r"compensation",
    ],
    "Account": [
        r"password", r"reset", r"profile", r"activate", r"deactivate",
        r"account closure", r"close.*account", r"username", r"email change",
    ],
}

HIGH_RISK_PATTERNS = {
    "Refund request": [r"refund"],
    "Subscription cancellation": [r"cancel.*subscription", r"subscription.*cancel"],
    "Account closure request": [r"close.*account", r"account closure", r"delete.*account"],
    "Compensation request": [r"compensation", r"goodwill", r"service credit"],
    "Escalation to management": [r"escalat", r"manager", r"management"],
}


def classify_intent_rule_based(query: str) -> str:
    q = query.lower()
    # Check memory-recall first, it takes precedence over department routing
    for pattern in INTENT_KEYWORDS["Memory"]:
        if re.search(pattern, q):
            return "Memory"
    scores = {}
    for intent, patterns in INTENT_KEYWORDS.items():
        if intent == "Memory":
            continue
        scores[intent] = sum(1 for p in patterns if re.search(p, q))
    best_intent = max(scores, key=scores.get)
    if scores[best_intent] == 0:
        return "Technical"  # safe default
    return best_intent


def detect_high_risk(query: str):
    q = query.lower()
    for reason, patterns in HIGH_RISK_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return True, reason
    return False, ""
