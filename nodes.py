"""
nodes.py
Implements every node used in the LangGraph workflow.

Task 3  -> intent_classification_node
Task 5  -> sales_agent_node, technical_agent_node, billing_agent_node, account_agent_node
Task 6  -> RAG retrieval is called from inside each agent node (see rag.py)
Task 7  -> memory_recall_node, save_memory_node (see memory.py)
Task 8  -> human_approval_node (uses LangGraph interrupt())
Task 9  -> supervisor_node
"""

from langgraph.types import interrupt

from state import SupportState
import memory
from rag import get_knowledge_base
from llm import llm_complete, classify_intent_rule_based, detect_high_risk

DEPARTMENT_NAME = {
    "Sales": "Sales Support",
    "Technical": "Technical Support",
    "Billing": "Billing Support",
    "Account": "Account Support",
}


def _log(state: SupportState, message: str) -> list:
    trace = list(state.get("trace", []))
    trace.append(message)
    return trace


# ---------------------------------------------------------------------------
# Task 3: Intent Classification node
# ---------------------------------------------------------------------------
def intent_classification_node(state: SupportState) -> SupportState:
    query = state["query"]

    fallback_intent = classify_intent_rule_based(query)
    raw = llm_complete(
        system_prompt=(
            "Classify the customer query into exactly one of: Sales, Technical, "
            "Billing, Account, Memory. Reply with only the single word."
        ),
        user_prompt=query,
        fallback=fallback_intent,
    )
    # Normalize whatever came back (real LLM or fallback) to one of the 5 labels.
    valid_labels = {"Sales", "Technical", "Billing", "Account", "Memory"}
    candidate = raw.strip().split()[0].strip(".,!").capitalize() if raw.strip() else fallback_intent
    intent = candidate if candidate in valid_labels else fallback_intent

    is_memory_query = intent == "Memory"
    department = "Memory Recall" if is_memory_query else DEPARTMENT_NAME[intent]

    return {
        **state,
        "intent": intent,
        "department": department,
        "is_memory_query": is_memory_query,
        "trace": _log(state, f"[Intent Classification] '{query}' -> intent={intent}, department={department}"),
    }


# ---------------------------------------------------------------------------
# Task 7: Memory recall node (handles "What was my previous issue?")
# ---------------------------------------------------------------------------
def memory_recall_node(state: SupportState) -> SupportState:
    customer_id = state["customer_id"]
    last_issue = memory.get_last_issue(customer_id)

    if last_issue:
        response = (
            f"Hi {state.get('customer_name', '')}, your previous support request was: "
            f"\"{last_issue['query']}\" (handled by {last_issue['department']} on "
            f"{last_issue['timestamp']}). Our response at the time was: \"{last_issue['response']}\""
        )
    else:
        response = (
            f"Hi {state.get('customer_name', '')}, I don't see any previous support "
            f"requests on file for you yet."
        )

    return {
        **state,
        "draft_response": response,
        "final_response": response,
        "requires_approval": False,
        "approval_status": "not_required",
        "trace": _log(state, "[Memory Recall] Retrieved conversation history from SQLite memory.db"),
    }


# ---------------------------------------------------------------------------
# Task 5 + Task 6: Specialized department agents (each performs RAG retrieval)
# ---------------------------------------------------------------------------
def _run_agent(state: SupportState, department_key: str, agent_label: str) -> SupportState:
    query = state["query"]
    kb = get_knowledge_base()
    results = kb.retrieve(query, department=department_key, top_k=3)

    context_chunks = [r[0] for r in results]
    sources = sorted(set(r[1] for r in results))

    is_high_risk, risk_reason = detect_high_risk(query)

    context_text = "\n".join(f"- {c}" for c in context_chunks) if context_chunks else "(no matching documents found)"

    # The approval status line ("pending" / "approved" / ...) is NOT baked into
    # the draft any more. The draft is written before a supervisor decides, so
    # a "pending" sentence here used to survive into the final reply even after
    # approval. supervisor_node now adds the status line that matches the
    # actual decision.
    fallback_draft = (
        f"Thank you for contacting {agent_label}. Based on our records:\n{context_text}"
    ).strip()

    draft = llm_complete(
        system_prompt=f"You are a {agent_label} agent for ABC Technologies. Use the provided context to answer helpfully and concisely.",
        user_prompt=f"Customer query: {query}\n\nContext:\n{context_text}",
        fallback=fallback_draft,
    )

    return {
        **state,
        "draft_response": draft,
        "retrieved_context": context_chunks,
        "retrieved_sources": sources,
        "is_high_risk": is_high_risk,
        "risk_reason": risk_reason,
        "requires_approval": is_high_risk,
        "approval_status": "pending" if is_high_risk else "not_required",
        "trace": _log(
            state,
            f"[{agent_label}] RAG retrieved {len(context_chunks)} chunk(s) from {sources}. "
            f"High-risk={is_high_risk} ({risk_reason or 'n/a'}).",
        ),
    }


def sales_agent_node(state: SupportState) -> SupportState:
    return _run_agent(state, "Sales", "Sales Support")


def technical_agent_node(state: SupportState) -> SupportState:
    return _run_agent(state, "Technical", "Technical Support")


def billing_agent_node(state: SupportState) -> SupportState:
    return _run_agent(state, "Billing", "Billing Support")


def account_agent_node(state: SupportState) -> SupportState:
    return _run_agent(state, "Account", "Account Support")


# ---------------------------------------------------------------------------
# Task 8: Human-in-the-loop approval node
# ---------------------------------------------------------------------------
APPROVE_WORDS = {"approve", "approved"}
REJECT_WORDS = {"reject", "rejected"}


def normalize_decision(raw) -> tuple:
    """Turn whatever the supervisor sent back into (approval_status, comment).

    Accepted inputs:
      - {"decision": "approve" | "reject", "comment": "..."}  (case-insensitive,
        surrounding spaces ignored, "approved"/"rejected" also accepted)
      - a bare string such as "approve" or "REJECT"

    Anything else (missing key, empty, None, numbers, "maybe", ...) returns
    "invalid_decision". It is never treated as an approval, and the customer
    is not told the request was declined either.
    """
    comment = ""
    if isinstance(raw, dict):
        value = raw.get("decision")
        comment = raw.get("comment") or ""
        comment = comment.strip() if isinstance(comment, str) else str(comment)
    else:
        value = raw

    word = value.strip().lower() if isinstance(value, str) else ""
    if word in APPROVE_WORDS:
        return "approved", comment
    if word in REJECT_WORDS:
        return "rejected", comment
    return "invalid_decision", comment


def human_approval_node(state: SupportState) -> SupportState:
    """
    Pauses the graph using LangGraph's interrupt() when a request is
    high-risk (refund, cancellation, account closure, compensation,
    escalation). Execution resumes when the calling code provides a
    Command(resume=<decision>) — see main.py for the demo of resuming with
    an "approve" / "reject" decision plus an optional comment.
    """
    payload = {
        "type": "approval_request",
        "customer_id": state["customer_id"],
        "customer_name": state.get("customer_name"),
        "query": state["query"],
        "department": state["department"],
        "risk_reason": state.get("risk_reason"),
        "draft_response": state.get("draft_response"),
        "instructions": (
            "Reply with {'decision': 'approve'|'reject', 'comment': str} "
            "(case-insensitive). Any other value leaves the request under review."
        ),
    }

    raw_decision = interrupt(payload)  # graph execution pauses here until resumed

    approval_status, comment = normalize_decision(raw_decision)

    if approval_status == "invalid_decision":
        log_line = (
            f"[Human Approval] Unrecognised supervisor decision {raw_decision!r}; "
            "request left under review (not approved, not declined)."
        )
    else:
        log_line = (
            f"[Human Approval] Supervisor decision: {approval_status.upper()}"
            + (f" — comment: {comment}" if comment else "")
        )

    return {
        **state,
        "approval_status": approval_status,
        "approver_comment": comment,
        "trace": _log(state, log_line),
    }


# ---------------------------------------------------------------------------
# Task 9: Supervisor agent (validates / improves the response before sending)
# ---------------------------------------------------------------------------
def supervisor_node(state: SupportState) -> SupportState:
    draft = state.get("draft_response", "")
    approval_status = state.get("approval_status", "not_required")

    if approval_status == "rejected":
        final = (
            f"We have reviewed your request regarding \"{state['query']}\" and are unable to "
            f"approve it at this time."
            + (f" Reason: {state.get('approver_comment')}" if state.get("approver_comment") else "")
            + " Please contact support for further assistance."
        )
        notes = "Supervisor rejected the request; replaced draft with a polite decline."
    elif approval_status == "invalid_decision":
        final = (
            f"Thank you for your patience. Your request regarding \"{state['query']}\" "
            "is still under review by a supervisor, and we will follow up with a decision shortly."
        )
        notes = "Supervisor decision was missing or unrecognised; request kept under review."
    else:
        notes_parts = []
        final = draft
        if approval_status == "approved":
            notes_parts.append("Supervisor approved the high-risk request.")
        if not final.strip():
            final = "Thank you for reaching out. Our team is reviewing your request and will follow up shortly."
            notes_parts.append("Draft was empty; substituted a safe default response.")
        else:
            notes_parts.append("Draft response validated and approved for sending.")
        notes = " ".join(notes_parts)

    final = llm_complete(
        system_prompt="You are a Supervisor agent. Lightly polish the draft response for tone and clarity without changing facts. Keep it concise.",
        user_prompt=final,
        fallback=final,
    )

    # Added after polishing so the status line always matches the real decision.
    if approval_status == "approved":
        final = f"{final}\n\nYour request has been approved by a supervisor and is now being processed."

    return {
        **state,
        "final_response": final,
        "supervisor_notes": notes,
        "trace": _log(state, f"[Supervisor] {notes}"),
    }


# ---------------------------------------------------------------------------
# Task 7: Save memory node — persists the final interaction to SQLite
# ---------------------------------------------------------------------------
def save_memory_node(state: SupportState) -> SupportState:
    if not state.get("is_memory_query"):
        memory.save_interaction(
            customer_id=state["customer_id"],
            customer_name=state.get("customer_name", ""),
            query=state["query"],
            intent=state.get("intent", ""),
            department=state.get("department", ""),
            response=state.get("final_response", ""),
        )
        msg = "[Memory Save] Interaction persisted to memory.db (SQLite)."
    else:
        msg = "[Memory Save] Skipped — this was a memory-recall query, not a new issue."

    return {**state, "trace": _log(state, msg)}
