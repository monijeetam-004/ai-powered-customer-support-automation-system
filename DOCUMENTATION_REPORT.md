# Documentation Report
## AI-Powered Customer Support Automation System — Built with LangGraph

### 1. Project Overview
This project implements the AI-powered customer support automation system
described in the assignment for ABC Technologies, using LangGraph to
orchestrate intent classification, department routing, RAG-backed
specialized agents, SQLite-based conversation memory, human-in-the-loop
approval for high-risk requests, and a supervisor agent that validates the
final response.

### 2. Task 1 — Workflow Design
See `workflow_diagram.png` (rendered) and `workflow_diagram.mmd` (mermaid
source). The graph begins at `intent_classification`, conditionally routes to
one of four department agents or directly to `memory_recall`, and converges
through an optional `human_approval` checkpoint into `supervisor`, then
`save_memory`, then `END`.

### 3. Task 2 — State Structure (`state.py`)
`SupportState` is a `TypedDict` carrying: customer identity (`customer_id`,
`customer_name`), query details (`query`, `intent`, `department`), risk flags
(`is_high_risk`, `risk_reason`), RAG output (`retrieved_context`,
`retrieved_sources`), memory data (`conversation_history`, `is_memory_query`),
response fields (`draft_response`, `supervisor_notes`, `final_response`),
and approval fields (`requires_approval`, `approval_status`,
`approver_comment`). A `trace` list records every node's action for
demonstration/audit purposes.

### 4. Task 3 — Intent Classification (`nodes.py::intent_classification_node`)
Classifies each query into `Sales`, `Technical`, `Billing`, `Account`, or
`Memory`. The classifier is implemented twice:
- A deterministic keyword/regex-based classifier (`llm.py::classify_intent_rule_based`)
  used by default so the project runs without any API key.
- An LLM-backed path (`llm.py::llm_complete`) that is used automatically if
  `ANTHROPIC_API_KEY` is set, with the rule-based result as a safe fallback.

### 5. Task 4 — Conditional Routing (`graph.py`)
Two conditional edges implement routing:
- `route_by_intent`: sends the state to the matching department agent or to
  `memory_recall`.
- `route_after_agent`: sends the state to `human_approval` if
  `requires_approval` is `True`, otherwise straight to `supervisor`.

### 6. Task 5 — Specialized Department Agents (`nodes.py`)
`sales_agent_node`, `technical_agent_node`, `billing_agent_node`, and
`account_agent_node` each call a shared `_run_agent()` helper that performs
RAG retrieval scoped to the department's relevant documents, detects
high-risk request types, and produces a draft response.

### 7. Task 6 — RAG Pipeline (`rag.py`)
The four knowledge base documents (Company Policy, Pricing Guide, Technical
Manual, FAQ) are loaded and chunked by paragraph. A `TfidfVectorizer` +
cosine similarity retriever returns the top-k most relevant chunks, first
restricting the candidate pool to documents mapped to the requesting
department (`DEPARTMENT_DOC_MAP`) for higher precision.

### 8. Task 7 — SQLite Memory (`memory.py`)
A `conversations` table in `memory.db` stores every resolved interaction
(`customer_id`, `timestamp`, `query`, `intent`, `department`, `response`).
`get_last_issue()` powers the "What was my previous support issue?" demo
query by retrieving the customer's most recent non-memory interaction.

### 9. Task 8 — Human-in-the-Loop Approval (`nodes.py::human_approval_node`, `graph.py`)
High-risk categories — refund requests, subscription cancellation, account
closure, compensation requests, and escalation to management — are detected
by `llm.py::detect_high_risk()`. When detected, the graph routes to
`human_approval`, which calls LangGraph's `interrupt()` to pause execution
and surface the request for supervisor review. A `SqliteSaver` checkpointer
(`checkpoints.db`) persists the paused state so the graph can be resumed
later (even after a process restart) with `Command(resume={"decision": "approve"|"reject", "comment": ...})`.

### 10. Task 9 — Supervisor Agent (`nodes.py::supervisor_node`)
Runs after every department agent (and after human approval, when
triggered). If the human approver rejected the request, the supervisor
rewrites the final response into a polite decline; otherwise it validates
and lightly polishes the draft response before it is sent to the customer.

### 11. Task 10 — Demonstration (`main.py`, `demo_output.log`)
`main.py` runs the five sample queries end-to-end and prints the full trace
and final response for each. Query 4 (refund) demonstrates the
interrupt → simulated supervisor decision → resume flow. Query 5
demonstrates memory recall pulling the prior billing interaction from
`memory.db`. See `demo_output.log` for a captured run.

### 12. Mapping to Submission Checklist
| Deliverable | File |
|---|---|
| Source Code | all `.py` files in this folder |
| README.md | `README.md` |
| Workflow Diagram | `workflow_diagram.png`, `workflow_diagram.mmd` |
| Screenshots / Execution Evidence | `demo_output.log` (text capture of full run) |
| SQLite Memory File/Schema | `memory.db` (generated on run) + schema in `memory.py` |
| Task Output Evidence | `demo_output.log` shows agent routing, RAG retrieval, human-in-the-loop approval, memory storage & recall, and final response generation for each of the 5 queries |
