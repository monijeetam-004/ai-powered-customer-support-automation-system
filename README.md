# AI-Powered Customer Support Automation System (LangGraph)

A LangGraph-based customer support automation system for ABC Technologies that
classifies customer intent, routes queries to specialized department agents,
retrieves answers from company documents via RAG, remembers customers across
sessions using SQLite, and requires human supervisor approval for high-risk
requests (refunds, cancellations, account closures, compensation, escalations)
before a final response is sent.

## 1. Project Structure

```
support_system/
├── state.py              # Task 2 - SupportState schema
├── memory.py             # Task 7 - SQLite conversation memory (memory.db)
├── rag.py                # Task 6 - TF-IDF based RAG retrieval pipeline
├── llm.py                # Intent classifier + optional real-LLM hook
├── nodes.py              # Tasks 3,5,6,7,8,9 - all graph node implementations
├── graph.py              # Tasks 1,4 - LangGraph workflow + conditional routing
├── main.py               # Task 10 - demo runner for the 5 sample queries
├── generate_diagram.py   # builds workflow_diagram.png
├── workflow_diagram.png  # rendered workflow diagram (deliverable)
├── workflow_diagram.mmd  # mermaid source for the diagram
├── knowledge_base/
│   ├── company_policy.txt
│   ├── pricing_guide.txt
│   ├── technical_manual.txt
│   └── faq.txt
├── demo_output.log        # captured run of all 5 demo queries
├── memory.db               # SQLite memory created on first run
├── checkpoints.db          # LangGraph checkpoint DB (powers interrupt/resume)
├── requirements.txt
└── DOCUMENTATION_REPORT.md
```

## 2. Setup

```bash
pip install -r requirements.txt
```

By default, the system runs **fully offline** using a deterministic
rule-based classifier/responder so it can be graded without any API key.
To use real Claude responses for classification, agent replies, and the
supervisor's polishing step, just export a key before running — no code
changes needed:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## 3. Run the Demo

```bash
python3 main.py
```

This runs the five sample queries from the assignment in order:

| # | Query | Expected Path |
|---|-------|----------------|
| 1 | "What are the pricing plans available for your software?" | Sales |
| 2 | "I forgot my account password." | Account |
| 3 | "My application crashes whenever I upload a file." | Technical Support |
| 4 | "I need a refund for my annual subscription." | Billing — human approval required |
| 5 | "What was my previous support issue?" | Memory recall (no department routing) |

For Query 4, the graph pauses with LangGraph's `interrupt()` and waits for a
supervisor decision. `main.py` simulates an "approve" decision and resumes the
graph with `Command(resume=...)`. In a real deployment this would be wired to
a supervisor dashboard/UI instead of being auto-simulated.

## 4. Regenerate the Workflow Diagram

```bash
python3 generate_diagram.py
```

## 5. Inspecting Stored Memory

```bash
python3 -c "import memory; [print(r) for r in memory.get_history('CUST-1001')]"
```

## 6. Notes on Design Choices

- **RAG**: Uses TF-IDF + cosine similarity (scikit-learn) over the 4 knowledge
  base text documents instead of an external embeddings API, so the whole
  pipeline runs offline with no extra API key. It can be swapped for a vector
  store (FAISS/Chroma + embeddings) without changing any node interfaces.
- **Memory**: A real SQLite database (`memory.db`) with a `conversations`
  table, queried directly with `sqlite3` (Task 7).
- **Human-in-the-loop**: Implemented using LangGraph's native `interrupt()` /
  `Command(resume=...)` mechanism with a `SqliteSaver` checkpointer
  (`checkpoints.db`), so the graph can pause indefinitely (e.g. while waiting
  on a real supervisor) and resume exactly where it left off (Task 8).
- **Supervisor agent**: Reviews/polishes the final response, and rewrites it
  into a polite decline if the human approver rejects the request (Task 9).
