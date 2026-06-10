from typing import List, Dict, Any
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    PINECONE_API_KEY,
    PINECONE_DENSE_INDEX,
    PINECONE_SPARSE_INDEX,
    EMBED_MODEL,
    TOP_K,
    ALPHA,
    SCORE_THRESHOLD,
)
from src.embed_upsert import BM25_PATH


class HybridRetriever:
    """
    Hybrid retriever: queries dense + sparse indexes in parallel,
    merges results with alpha weighting, applies refusal threshold
    based on raw dense score.
    """

    def __init__(self):
        print("Initializing hybrid retriever...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.dense_index  = self.pc.Index(PINECONE_DENSE_INDEX)
        self.sparse_index = self.pc.Index(PINECONE_SPARSE_INDEX)

        self.embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        self.bm25 = BM25Encoder()
        self.bm25.load(BM25_PATH)

        print("Hybrid retriever ready")

    def _query_dense(self, query: str, k: int) -> Dict[str, Dict]:
        vec = self.embedder.embed_query(query)
        res = self.dense_index.query(
            vector=vec,
            top_k=k,
            include_metadata=True,
        )
        return {
            match.id: {"score": match.score, "metadata": match.metadata}
            for match in res.matches
        }

    def _query_sparse(self, query: str, k: int) -> Dict[str, Dict]:
        sparse_vec = self.bm25.encode_queries(query)
        res = self.sparse_index.query(
            sparse_vector=sparse_vec,
            top_k=k,
            include_metadata=True,
        )
        return {
            match.id: {"score": match.score, "metadata": match.metadata}
            for match in res.matches
        }

    @staticmethod
    def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def retrieve(self, query: str, k: int = TOP_K, alpha: float = ALPHA) -> Dict[str, Any]:
        fetch_k = k * 3

        dense_hits  = self._query_dense(query, fetch_k)
        sparse_hits = self._query_sparse(query, fetch_k)

        raw_dense_scores = {i: h["score"] for i, h in dense_hits.items()}
        top_dense_raw = max(raw_dense_scores.values()) if raw_dense_scores else 0.0

        dense_norm  = self._normalize(raw_dense_scores)
        sparse_norm = self._normalize({i: h["score"] for i, h in sparse_hits.items()})

        all_ids = set(dense_hits) | set(sparse_hits)
        merged = {}
        for doc_id in all_ids:
            d = dense_norm.get(doc_id, 0.0)
            s = sparse_norm.get(doc_id, 0.0)
            merged[doc_id] = alpha * d + (1 - alpha) * s

        ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:k]

        documents, scores = [], []
        for doc_id, score in ranked:
            source = dense_hits.get(doc_id) or sparse_hits.get(doc_id)
            meta = dict(source["metadata"])
            text = meta.pop("text", "")
            documents.append(Document(page_content=text, metadata=meta))
            scores.append(score)

        refused = top_dense_raw < SCORE_THRESHOLD

        return {
            "documents": documents,
            "scores": scores,
            "refused": refused,
            "top_score": top_dense_raw,
            "merged_score": scores[0] if scores else 0.0,
        }


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "Who won the 1998 FIFA World Cup?",
        "How many goals did Maradona score in 1986?",
        "What was the Hand of God?",
        "Which team had the best goal difference in 2022?",
        "What is the recipe for chocolate cake?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        result = retriever.retrieve(query)

        print(f"Top dense score (refusal check): {result['top_score']:.3f}")
        print(f"Top merged score (ranking):      {result['merged_score']:.3f}")

        if result["refused"]:
            print(f"REFUSED — top dense score below threshold ({SCORE_THRESHOLD})")
            continue

        for i, (doc, score) in enumerate(zip(result["documents"], result["scores"]), 1):
            print(f"\n  [{i}] merged_score={score:.3f}  source={doc.metadata.get('source')}")
            print(f"      {doc.page_content[:200]}...")