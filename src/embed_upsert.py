import os
import json
import time
from typing import List
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    PINECONE_API_KEY,
    PINECONE_DENSE_INDEX,
    PINECONE_SPARSE_INDEX,
    EMBED_MODEL,
)
from src.ingest import build_corpus


BM25_PATH = "data/bm25_worldcup.json"
BATCH_SIZE = 100


def dedupe_by_doc_id(docs: List[Document]) -> List[Document]:
    """Keep only the first occurrence of each doc_id."""
    seen = set()
    unique = []
    for d in docs:
        doc_id = d.metadata.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique.append(d)
    return unique


def fit_and_save_bm25(texts: List[str]) -> BM25Encoder:
    """Fit BM25 on the corpus and persist the weights."""
    print(f"Fitting BM25 on {len(texts)} documents...")
    bm25 = BM25Encoder()
    bm25.fit(texts)
    bm25.dump(BM25_PATH)
    print(f"BM25 weights saved to {BM25_PATH}")
    return bm25


def upsert_dense(docs: List[Document], embedder: HuggingFaceEmbeddings, index):
    """Embed and upsert dense vectors in batches."""
    print(f"\nUpserting {len(docs)} dense vectors...")
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        texts = [d.page_content for d in batch]
        vectors = embedder.embed_documents(texts)

        records = []
        for doc, vec in zip(batch, vectors):
            metadata = {k: v for k, v in doc.metadata.items() if v is not None}
            metadata["text"] = doc.page_content
            records.append({
                "id": doc.metadata["doc_id"],
                "values": vec,
                "metadata": metadata,
            })

        index.upsert(vectors=records)
        print(f"  Dense batch {i // BATCH_SIZE + 1}: upserted {len(batch)} vectors")
        time.sleep(0.2)


def upsert_sparse(docs: List[Document], bm25: BM25Encoder, index):
    """Encode and upsert sparse vectors in batches."""
    print(f"\nUpserting {len(docs)} sparse vectors...")
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i + BATCH_SIZE]
        texts = [d.page_content for d in batch]
        sparse_vecs = bm25.encode_documents(texts)

        records = []
        for doc, sv in zip(batch, sparse_vecs):
            metadata = {k: v for k, v in doc.metadata.items() if v is not None}
            metadata["text"] = doc.page_content
            records.append({
                "id": doc.metadata["doc_id"],
                "sparse_values": sv,
                "metadata": metadata,
            })

        index.upsert(vectors=records)
        print(f"  Sparse batch {i // BATCH_SIZE + 1}: upserted {len(batch)} vectors")
        time.sleep(0.2)


def main():
    docs = build_corpus()
    docs = dedupe_by_doc_id(docs)
    print(f"\nAfter dedup: {len(docs)} unique documents")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    dense_index = pc.Index(PINECONE_DENSE_INDEX)
    sparse_index = pc.Index(PINECONE_SPARSE_INDEX)

    texts = [d.page_content for d in docs]
    bm25 = fit_and_save_bm25(texts)

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("(First run downloads ~90MB — subsequent runs are instant)")
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    upsert_dense(docs, embedder, dense_index)
    upsert_sparse(docs, bm25, sparse_index)

    print("\nDone. Verifying index stats...")
    print(f"Dense index:  {dense_index.describe_index_stats()}")
    print(f"Sparse index: {sparse_index.describe_index_stats()}")


if __name__ == "__main__":
    main()