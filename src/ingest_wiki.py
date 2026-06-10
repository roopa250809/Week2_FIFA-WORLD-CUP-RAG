import os
import re
import json
import wikipediaapi
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.wiki_sources import WIKIPEDIA_ARTICLES
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

WIKI_CACHE_DIR = "data/wiki_cache"
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

wiki = wikipediaapi.Wikipedia(
    user_agent="WorldCupRAG/1.0 (educational project)",
    language="en",
)


def _safe_filename(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_") + ".json"


def fetch_article(title: str) -> dict | None:
    """Fetch one article from Wikipedia, cached locally to avoid re-fetching."""
    cache_path = os.path.join(WIKI_CACHE_DIR, _safe_filename(title))
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    page = wiki.page(title)
    if not page.exists():
        print(f"  WARNING: Article not found: {title}")
        return None

    article = {
        "title": page.title,
        "url": page.fullurl,
        "summary": page.summary,
        "text": page.text,
    }

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)

    return article


def _clean_text(text: str) -> str:
    """Drop reference markers, normalize whitespace."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_wikipedia_articles() -> List[Document]:
    """
    Fetch each article, clean it, and split into semantic chunks.
    Returns LangChain Documents with rich metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_docs = []
    print(f"Fetching {len(WIKIPEDIA_ARTICLES)} Wikipedia articles...")

    for title in WIKIPEDIA_ARTICLES:
        article = fetch_article(title)
        if not article:
            continue

        cleaned = _clean_text(article["text"])
        chunks = splitter.split_text(cleaned)
        print(f"  OK {article['title']}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": "wikipedia",
                    "title": article["title"],
                    "url": article["url"],
                    "chunk_index": i,
                    "doc_id": f"wiki-{_safe_filename(title).replace('.json', '')}-{i}",
                }
            )
            all_docs.append(doc)

    return all_docs


if __name__ == "__main__":
    docs = load_wikipedia_articles()
    print(f"\nTotal Wikipedia chunks: {len(docs)}")
    print("\n--- Sample chunk ---")
    print(docs[0].page_content[:300], "...")
    print("\nMetadata:", docs[0].metadata)