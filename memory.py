"""
memory.py
Task 7: SQLite-based memory to store and retrieve customer conversation history.

Creates memory.db with a `conversations` table. Provides helper functions to
save a turn and to fetch the most recent turn(s) for a customer, which powers
the "What was my previous issue?" memory-recall use case.
"""

import sqlite3
from datetime import datetime
from typing import List, Dict, Optional

DB_PATH = "memory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id TEXT NOT NULL,
    customer_name TEXT,
    timestamp TEXT NOT NULL,
    query TEXT NOT NULL,
    intent TEXT,
    department TEXT,
    response TEXT
);
"""


def init_db(db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


def save_interaction(
    customer_id: str,
    customer_name: str,
    query: str,
    intent: str,
    department: str,
    response: str,
    db_path: str = DB_PATH,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO conversations
           (customer_id, customer_name, timestamp, query, intent, department, response)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            customer_id,
            customer_name,
            datetime.utcnow().isoformat(timespec="seconds"),
            query,
            intent,
            department,
            response,
        ),
    )
    conn.commit()
    conn.close()


def get_history(customer_id: str, limit: int = 10, db_path: str = DB_PATH) -> List[Dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM conversations WHERE customer_id = ?
           ORDER BY id DESC LIMIT ?""",
        (customer_id, limit),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_last_issue(customer_id: str, db_path: str = DB_PATH) -> Optional[Dict]:
    """Return the most recent non-memory interaction for a customer."""
    history = get_history(customer_id, limit=20, db_path=db_path)
    for row in history:
        if row["intent"] != "Memory":
            return row
    return None
