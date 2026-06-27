"""
main.py
Task 10: Demonstrates the system using the five sample customer queries.

Run:
    python3 main.py

Optional (to use a real Claude LLM for classification/agents/supervisor
instead of the offline rule-based fallback):
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 main.py
"""

import uuid
from langgraph.types import Command

import memory
from graph import build_graph

SEPARATOR = "=" * 80


def print_trace(result: dict):
    for line in result.get("trace", []):
        print("   ", line)


def run_query(app, customer_id: str, customer_name: str, query: str):
    print(SEPARATOR)
    print(f"CUSTOMER ({customer_name}): {query}")
    print(SEPARATOR)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "customer_id": customer_id,
        "customer_name": customer_name,
        "query": query,
        "trace": [],
    }

    result = app.invoke(initial_state, config=config)

    # If the graph paused for human approval, `__interrupt__` will be present.
    if "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        approval_request = interrupt_obj.value
        print("\n   *** HUMAN-IN-THE-LOOP CHECKPOINT ***")
        print(f"    A supervisor must approve this {approval_request['risk_reason']} request:")
        print(f"    Draft response: {approval_request['draft_response']}")

        # --- Simulated supervisor decision for the demo ---
        # In a real deployment this would come from a supervisor's dashboard.
        decision = {"decision": "approve", "comment": "Approved per 30-day refund policy."}
        print(f"    -> Supervisor decision: {decision['decision'].upper()} ({decision['comment']})")

        result = app.invoke(Command(resume=decision), config=config)

    print()
    print_trace(result)
    print()
    print("FINAL RESPONSE TO CUSTOMER:")
    print(f"   {result.get('final_response')}")
    print()
    return result


def main():
    memory.init_db()
    app = build_graph()

    customer_id = "CUST-1001"
    customer_name = "David"

    queries = [
        "What are the pricing plans available for your software?",      # Sales
        "I forgot my account password.",                                  # Account
        "My application crashes whenever I upload a file.",               # Technical
        "I need a refund for my annual subscription.",                    # Billing -> human approval
        "What was my previous support issue?",                            # Memory recall
    ]

    for i, q in enumerate(queries, start=1):
        print(f"\n### QUERY {i} ###")
        run_query(app, customer_id, customer_name, q)

    print(SEPARATOR)
    print("Full conversation history stored in memory.db for", customer_id)
    print(SEPARATOR)
    for row in memory.get_history(customer_id):
        print(row)


if __name__ == "__main__":
    main()
