"""
rag.py
Task 6: RAG pipeline over the company knowledge base documents.

Loads the four knowledge base documents, splits them into paragraph-level
chunks, and retrieves the most relevant chunks for a given query using
TF-IDF + cosine similarity. This keeps the project runnable fully offline
(no external embedding API key required) while still implementing a real
retrieval pipeline. The retriever can be swapped for an embeddings-based
vector store (e.g. FAISS + OpenAI/Anthropic embeddings) without changing
the rest of the graph.
"""

import os
import glob
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

DEPARTMENT_DOC_MAP = {
    "Sales": ["pricing_guide.txt", "faq.txt"],
    "Technical": ["technical_manual.txt", "faq.txt"],
    "Billing": ["company_policy.txt", "pricing_guide.txt"],
    "Account": ["faq.txt", "company_policy.txt"],
}


class KnowledgeBase:
    def __init__(self, kb_dir: str = KB_DIR):
        self.kb_dir = kb_dir
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self._load_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def _load_documents(self):
        for path in sorted(glob.glob(os.path.join(self.kb_dir, "*.txt"))):
            filename = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            # paragraph-level chunking
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for p in paragraphs:
                self.chunks.append(p)
                self.sources.append(filename)

    def retrieve(self, query: str, department: str = None, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Return list of (chunk_text, source_file, score) sorted by relevance.

        If `department` is provided, restrict the candidate pool to the
        documents relevant for that department first (improves precision),
        falling back to the full corpus if nothing scores above 0.
        """
        candidate_idxs = list(range(len(self.chunks)))
        if department and department in DEPARTMENT_DOC_MAP:
            allowed_docs = set(DEPARTMENT_DOC_MAP[department])
            filtered = [i for i in candidate_idxs if self.sources[i] in allowed_docs]
            if filtered:
                candidate_idxs = filtered

        query_vec = self.vectorizer.transform([query])
        sub_matrix = self.matrix[candidate_idxs]
        sims = cosine_similarity(query_vec, sub_matrix).flatten()

        ranked = sorted(zip(candidate_idxs, sims), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score > 0:
                results.append((self.chunks[idx], self.sources[idx], float(score)))
        return results


# Singleton knowledge base instance, loaded once per process.
_kb_instance = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
