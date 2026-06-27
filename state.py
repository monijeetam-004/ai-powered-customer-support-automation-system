"""
state.py
Task 2: State structure for the Customer Support Automation System.

The state object flows through every node of the LangGraph graph and carries
customer information, query details, retrieved RAG context, approval status,
and the response payload.
"""

from typing import TypedDict, List, Dict, Optional


class SupportState(TypedDict, total=False):
    # --- Customer information ---
    customer_id: str
    customer_name: str

    # --- Query details ---
    query: str
    intent: str                # Sales | Technical | Billing | Account | Memory
    department: str            # Sales Support | Technical Support | Billing Support |
                                # Account Support | Memory Recall
    is_high_risk: bool         # True for refund / cancellation / closure / compensation / escalation
    risk_reason: str           # which high-risk category triggered approval

    # --- RAG ---
    retrieved_context: List[str]
    retrieved_sources: List[str]

    # --- Memory ---
    conversation_history: List[Dict]
    is_memory_query: bool

    # --- Draft / Final response ---
    draft_response: str
    supervisor_notes: str
    final_response: str

    # --- Human-in-the-loop ---
    requires_approval: bool
    approval_status: str       # "not_required" | "pending" | "approved" | "rejected"
    approver_comment: Optional[str]

    # --- Trace (for screenshots / demo) ---
    trace: List[str]
