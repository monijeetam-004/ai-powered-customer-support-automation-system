"""
Regression tests for the human-in-the-loop approval path.

Bugs these tests reproduce (all present before this fix):
  1. After a supervisor APPROVED a refund, the final reply (and the copy saved
     to memory) still said the request was "flagged for supervisor approval".
  2. A resume payload with no "decision" key was treated as a REJECTION,
     because the code defaulted to "approved" but only accepted "approve".
  3. "approved", "Approve", " APPROVE " etc. were also treated as rejections.
"""

import pytest

import memory

PENDING_TEXT = "flagged for supervisor approval"
REFUND = "I need a refund for my annual subscription."


# --- Bug 1: approved requests must not still read as "pending" --------------

def test_approved_refund_final_response_is_not_pending(run_query):
    result = run_query(REFUND, decision={"decision": "approve", "comment": "OK"})

    assert result["approval_status"] == "approved"
    assert PENDING_TEXT not in result["final_response"]
    assert "approved" in result["final_response"].lower()


def test_approved_refund_is_saved_to_memory_without_pending_text(run_query):
    run_query(REFUND, decision={"decision": "approve"}, customer_id="CUST-MEM")

    saved = memory.get_last_issue("CUST-MEM")
    assert saved is not None
    assert PENDING_TEXT not in saved["response"]


def test_memory_recall_after_approval_does_not_say_pending(run_query):
    run_query(REFUND, decision={"decision": "approve"}, customer_id="CUST-RECALL")
    recall = run_query("What was my previous support issue?", customer_id="CUST-RECALL")

    assert recall["intent"] == "Memory"
    assert REFUND in recall["final_response"]
    assert PENDING_TEXT not in recall["final_response"]


# --- Bugs 2 + 3: explicit, case-insensitive decision handling ---------------

@pytest.mark.parametrize(
    "decision",
    [
        {"decision": "approve"},
        {"decision": "Approve"},
        {"decision": "  APPROVE  "},
        {"decision": "approved"},
        "approve",  # a bare string resume value is accepted too
    ],
)
def test_approve_variants_are_approved(run_query, decision):
    result = run_query(REFUND, decision=decision)
    assert result["approval_status"] == "approved"


@pytest.mark.parametrize(
    "decision",
    [{"decision": "reject"}, {"decision": "Rejected"}, "REJECT"],
)
def test_reject_variants_are_rejected_with_polite_decline(run_query, decision):
    result = run_query(REFUND, decision=decision)

    assert result["approval_status"] == "rejected"
    assert "unable to approve" in result["final_response"]


@pytest.mark.parametrize(
    "decision",
    [
        {"comment": "looks fine"},     # missing "decision" key (was silently rejected)
        {"decision": ""},
        {"decision": "maybe"},
        {"decision": None},
        {"decision": 1},
        "yes",
    ],
)
def test_missing_or_invalid_decision_is_neither_approved_nor_declined(run_query, decision):
    """An unclear supervisor answer must not approve a high-risk request, and
    must not tell the customer it was declined either. It stays under review."""
    result = run_query(REFUND, decision=decision)

    assert result["approval_status"] == "invalid_decision"
    final = result["final_response"].lower()
    assert "has been approved" not in final
    assert "unable to approve" not in final
    assert "review" in final


def test_empty_resume_keeps_the_request_paused(run_query):
    """LangGraph itself ignores Command(resume={}) and leaves the graph waiting
    at the approval step. That is safe: nothing is approved or declined."""
    result = run_query(REFUND, decision={})

    assert "__interrupt__" in result
    assert result["approval_status"] == "pending"
    assert "final_response" not in result


def test_rejection_comment_is_kept_as_the_reason(run_query):
    result = run_query(REFUND, decision={"decision": "reject", "comment": "Outside 30 days."})
    assert "Outside 30 days." in result["final_response"]


# --- The graph still pauses, and normal requests are unaffected -------------

def test_high_risk_request_pauses_for_approval(run_query):
    result = run_query(REFUND, decision=None)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["risk_reason"] == "Refund request"


def test_low_risk_request_skips_approval(run_query):
    result = run_query("What are the pricing plans available for your software?")

    assert "__interrupt__" not in result
    assert result["approval_status"] == "not_required"
    assert PENDING_TEXT not in result["final_response"]
    assert "approved" not in result["final_response"].lower()


# --- Unit tests for the normaliser itself ------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"decision": "approve"}, ("approved", "")),
        ({"decision": "REJECTED", "comment": " too late "}, ("rejected", "too late")),
        ("approve", ("approved", "")),
        ({"comment": "x"}, ("invalid_decision", "x")),
        (None, ("invalid_decision", "")),
        (42, ("invalid_decision", "")),
    ],
)
def test_normalize_decision(raw, expected):
    from nodes import normalize_decision

    assert normalize_decision(raw) == expected
