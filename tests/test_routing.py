"""
Regression tests for rule-based intent routing (llm.classify_intent_rule_based).

Bug reproduced before this fix: the Memory patterns were so broad (a bare
"history", and "what was my") that billing and pricing questions were routed
to memory recall instead of the correct department, e.g.
  "Can I see my billing history?"  -> Memory   (should be Billing)
  "What was my plan price?"        -> Memory   (should be Sales)

These tests are self-contained (no LangGraph run, no databases) and always
use the offline classifier.
"""

import os
import sys

import pytest

os.environ.pop("ANTHROPIC_API_KEY", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import classify_intent_rule_based  # noqa: E402
from graph import route_by_intent  # noqa: E402


# Misroutes that this change fixes.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("Can I see my billing history?", "Billing"),
        ("Show my payment history", "Billing"),
        ("What was my invoice amount last month?", "Billing"),
        ("What was my plan price?", "Sales"),
    ],
)
def test_department_questions_are_not_sent_to_memory(query, expected):
    assert classify_intent_rule_based(query) == expected


# Real memory-recall questions must still go to Memory.
@pytest.mark.parametrize(
    "query",
    [
        "What was my previous support issue?",  # demo query 5
        "What was my previous issue?",
        "What did I ask last time?",
        "Show my support history",
        "Remind me of my last ticket",
        "What did I report earlier?",
    ],
)
def test_memory_recall_questions_still_route_to_memory(query):
    assert classify_intent_rule_based(query) == "Memory"


# The five demo queries from main.py keep their documented routes.
@pytest.mark.parametrize(
    "query, expected",
    [
        ("What are the pricing plans available for your software?", "Sales"),
        ("I forgot my account password.", "Account"),
        ("My application crashes whenever I upload a file.", "Technical"),
        ("I need a refund for my annual subscription.", "Billing"),
        ("What was my previous support issue?", "Memory"),
    ],
)
def test_demo_queries_keep_their_routes(query, expected):
    assert classify_intent_rule_based(query) == expected


@pytest.mark.parametrize(
    "intent, node",
    [
        ("Sales", "sales_agent"),
        ("Technical", "technical_agent"),
        ("Billing", "billing_agent"),
        ("Account", "account_agent"),
        ("Memory", "memory_recall"),
        ("Unknown", "technical_agent"),  # documented safe default
    ],
)
def test_route_by_intent_maps_each_intent_to_its_node(intent, node):
    assert route_by_intent({"intent": intent}) == node
