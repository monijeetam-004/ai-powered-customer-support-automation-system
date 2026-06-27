"""
graph.py
Task 1: LangGraph workflow design for the Customer Support Automation System.
Task 4: Conditional routing to the appropriate support agent.

Workflow:

                         +------------------------+
                         | intent_classification |
                         +------------------------+
                                    |
                  -------------------------------------------
                  |        |          |          |          |
               (Sales) (Technical) (Billing) (Account)   (Memory)
                  v        v          v          v          |
             sales_agent technical_agent billing_agent account_agent
                  \\         |           |          /        |
                   \\        |           |         /         |
                    -------------------------------          |
                                  |                           |
                       requires_approval? --no--> supervisor  |
                              |yes                    ^       |
                              v                        \\      |
                       human_approval ------------------+      |
                                                                v
                                                          save_memory --> END
                                                                ^
                                                                |
                                              (memory path bypasses agents/approval)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from state import SupportState
from nodes import (
    intent_classification_node,
    memory_recall_node,
    sales_agent_node,
    technical_agent_node,
    billing_agent_node,
    account_agent_node,
    human_approval_node,
    supervisor_node,
    save_memory_node,
)


# --- Task 4: Conditional routing functions -----------------------------------

def route_by_intent(state: SupportState) -> str:
    """Routes from intent_classification to the correct department agent,
    or straight to memory_recall for history lookups."""
    intent = state["intent"]
    mapping = {
        "Sales": "sales_agent",
        "Technical": "technical_agent",
        "Billing": "billing_agent",
        "Account": "account_agent",
        "Memory": "memory_recall",
    }
    return mapping.get(intent, "technical_agent")


def route_after_agent(state: SupportState) -> str:
    """After any department agent runs, decide whether the request needs
    human-in-the-loop approval before reaching the supervisor."""
    return "human_approval" if state.get("requires_approval") else "supervisor"


def build_graph(checkpoint_path: str = "checkpoints.db"):
    graph = StateGraph(SupportState)

    # --- Nodes ---
    graph.add_node("intent_classification", intent_classification_node)
    graph.add_node("memory_recall", memory_recall_node)
    graph.add_node("sales_agent", sales_agent_node)
    graph.add_node("technical_agent", technical_agent_node)
    graph.add_node("billing_agent", billing_agent_node)
    graph.add_node("account_agent", account_agent_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("save_memory", save_memory_node)

    # --- Edges ---
    graph.add_edge(START, "intent_classification")

    graph.add_conditional_edges(
        "intent_classification",
        route_by_intent,
        {
            "sales_agent": "sales_agent",
            "technical_agent": "technical_agent",
            "billing_agent": "billing_agent",
            "account_agent": "account_agent",
            "memory_recall": "memory_recall",
        },
    )

    for agent in ["sales_agent", "technical_agent", "billing_agent", "account_agent"]:
        graph.add_conditional_edges(
            agent,
            route_after_agent,
            {"human_approval": "human_approval", "supervisor": "supervisor"},
        )

    graph.add_edge("human_approval", "supervisor")
    graph.add_edge("supervisor", "save_memory")
    graph.add_edge("memory_recall", "save_memory")
    graph.add_edge("save_memory", END)

    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled
