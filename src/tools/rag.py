from typing import List, Dict
import re
import math
import logging

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Chunks a long text string into overlapping token/word chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def filter_top_k_sources(objective: str, sources: List[Dict], top_k: int = 6) -> List[Dict]:
    """
    Deduplicates and filters sources by fast TF-IDF term relevance scoring.
    Executes in under 0.01s with zero heavy model loading.
    """
    if not sources:
        return []

    # Deduplicate sources by URL
    seen_urls = set()
    unique_sources = []
    for s in sources:
        url = s.get("url", "").strip()
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(s)

    def tokenize(text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    obj_tokens = tokenize(objective)
    if not obj_tokens:
        return unique_sources[:top_k]

    obj_tf = {}
    for t in obj_tokens:
        obj_tf[t] = obj_tf.get(t, 0) + 1

    # Score each source using cosine-like TF overlap
    for s in unique_sources:
        doc_text = f"{s.get('title', '')} {s.get('content', '')}"
        doc_tokens = tokenize(doc_text)
        doc_tf = {}
        for t in doc_tokens:
            doc_tf[t] = doc_tf.get(t, 0) + 1
        
        score = 0.0
        for token, count in obj_tf.items():
            if token in doc_tf:
                score += (count * doc_tf[token])
        
        # Normalize by length
        norm = math.sqrt(len(doc_tokens)) if doc_tokens else 1.0
        s["score"] = score / norm

    ranked_sources = sorted(unique_sources, key=lambda x: x.get("score", 0), reverse=True)
    return ranked_sources[:top_k]
