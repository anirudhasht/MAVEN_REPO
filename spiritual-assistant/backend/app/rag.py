import os
import re
from typing import List, Tuple
from .config import KNOWLEDGE_DIR

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return set(w.lower() for w in _WORD_RE.findall(text))


def _iter_text_files(root_dir: str) -> List[Tuple[str, str]]:
    documents: List[Tuple[str, str]] = []
    if not os.path.isdir(root_dir):
        return documents
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".txt", ".md")):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        documents.append((fpath, f.read()))
                except Exception:
                    continue
    return documents


class SimpleRetriever:
    def __init__(self, root_dir: str = KNOWLEDGE_DIR) -> None:
        self.root_dir = root_dir
        self.docs: List[Tuple[str, str]] = _iter_text_files(root_dir)
        self.doc_tokens: List[set[str]] = [_tokenize(text) for _, text in self.docs]

    def refresh(self) -> None:
        self.docs = _iter_text_files(self.root_dir)
        self.doc_tokens = [_tokenize(text) for _, text in self.docs]

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        if not self.docs:
            return []
        q_tokens = _tokenize(query)
        scored: List[Tuple[float, int]] = []
        for idx, tokens in enumerate(self.doc_tokens):
            if not tokens:
                continue
            overlap = len(q_tokens & tokens)
            score = overlap / (len(q_tokens) + 1e-9)
            if score > 0:
                scored.append((score, idx))
        scored.sort(reverse=True)
        selected = [self.docs[i][1] for _, i in scored[:top_k]]
        return selected


retriever = SimpleRetriever()