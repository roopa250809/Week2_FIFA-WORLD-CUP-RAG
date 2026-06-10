import re
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieve import HybridRetriever
from src.generate import AnswerGenerator, REFUSAL_RESPONSE
from src.config import ANTHROPIC_API_KEY, LLM_MODEL


class RAGState(TypedDict, total=False):
    """State that flows through every node in the graph."""
    query: str
    rewritten_query: str
    documents: List[Document]
    scores: List[float]
    top_score: float
    refused: bool
    refused_reason: Optional[str]
    answer: str
    citations: List[Dict[str, Any]]


_retriever: Optional[HybridRetriever] = None
_generator: Optional[AnswerGenerator] = None
_rewriter: Optional[ChatAnthropic] = None

REWRITE_SYSTEM = """You are a query rewriter for a FIFA World Cup search engine.
Convert the user's question into an optimal search query by:
- Replacing ordinal references with years (e.g. "2nd World Cup" → "1934 FIFA World Cup", "first" → "1930", "third" → "1938")
- Expanding abbreviations and nicknames (e.g. "R9" → "Ronaldo", "the GOAT" → "Lionel Messi")
- When a question asks how many goals a player scored "in [year]" with no other qualifier, add "FIFA World Cup" to make the scope explicit (e.g. "goals in 2022" → "goals in 2022 FIFA World Cup")
- Keeping the query concise and factual

Return ONLY the rewritten query, no explanation."""

ORDINAL_TO_YEAR = {
    "1st": "1930", "first": "1930",
    "2nd": "1934", "second": "1934",
    "3rd": "1938", "third": "1938",
    "4th": "1950", "fourth": "1950",
    "5th": "1954", "fifth": "1954",
    "6th": "1958", "sixth": "1958",
    "7th": "1962", "seventh": "1962",
    "8th": "1966", "eighth": "1966",
    "9th": "1970", "ninth": "1970",
    "10th": "1974", "tenth": "1974",
    "11th": "1978", "eleventh": "1978",
    "12th": "1982", "twelfth": "1982",
    "13th": "1986",
    "14th": "1990",
    "15th": "1994",
    "16th": "1998",
    "17th": "2002",
    "18th": "2006",
    "19th": "2010",
    "20th": "2014",
    "21st": "2018",
    "22nd": "2022",
}


def _get_rewriter() -> ChatAnthropic:
    global _rewriter
    if _rewriter is None:
        _rewriter = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
            max_tokens=80,
        )
    return _rewriter


CORPUS_MIN_YEAR = 1930
CORPUS_MAX_YEAR = 2022


PLAYER_GOAL_PATTERN = re.compile(
    r"\b(goals?|scored?|hat.?trick)\b.{0,40}\b(19\d{2}|20[01]\d|202[0-2])\b"
    r"|"
    r"\b(19\d{2}|20[01]\d|202[0-2])\b.{0,40}\b(goals?|scored?)\b",
    re.IGNORECASE,
)


_EDITION_ORDINAL_RE = re.compile(
    r'\b(?:the\s+)?(' + '|'.join(re.escape(k) for k in ORDINAL_TO_YEAR.keys()) + r')\s+(?:fifa\s+)?world\s+cup',
    re.IGNORECASE,
)


def _apply_ordinal_substitution(query: str) -> str:
    """
    Deterministic ordinal → year replacement, but only when the ordinal
    refers to a tournament edition (e.g. 'the 22nd World Cup'), not a
    finishing position (e.g. 'finished third').
    """
    m = _EDITION_ORDINAL_RE.search(query)
    if not m:
        return query
    ordinal_found = m.group(1).lower()
    year = ORDINAL_TO_YEAR.get(ordinal_found)
    if not year:
        return query
    return re.sub(re.escape(ordinal_found), year, query, count=1, flags=re.IGNORECASE)


def _out_of_scope_year(query: str) -> bool:
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", query)]
    return any(y < CORPUS_MIN_YEAR or y > CORPUS_MAX_YEAR for y in years)


def rewrite_node(state: RAGState) -> Dict[str, Any]:
    query = state["query"]

    if _out_of_scope_year(query):
        print(f"[rewrite_node] out-of-scope year detected — refusing")
        return {"rewritten_query": query, "refused": True, "top_score": 0.0}

    has_ordinal    = bool(_EDITION_ORDINAL_RE.search(query))
    has_goal_query = bool(PLAYER_GOAL_PATTERN.search(query))

    if not has_ordinal and not has_goal_query:
        print(f"[rewrite_node] no rewrite needed")
        return {"rewritten_query": query}

    # Ordinals: substitute programmatically — never trust the LLM for numeric mappings
    rewritten = _apply_ordinal_substitution(query) if has_ordinal else query

    # Player-goal queries: call LLM to add explicit "FIFA World Cup" scope
    if has_goal_query:
        llm = _get_rewriter()
        response = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=rewritten),
        ])
        rewritten = response.content.strip()

    print(f"[rewrite_node] '{query}' -> '{rewritten}'")
    return {"rewritten_query": rewritten}


def route_after_rewrite(state: RAGState) -> str:
    return "refuse" if state.get("refused") else "retrieve"


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_generator() -> AnswerGenerator:
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator


def retrieve_node(state: RAGState) -> Dict[str, Any]:
    """Node 2: hybrid retrieval using the (possibly rewritten) query."""
    search_query = state.get("rewritten_query") or state["query"]
    print(f"[retrieve_node] query: {search_query}")
    retriever = _get_retriever()
    result = retriever.retrieve(search_query)
    print(f"[retrieve_node] top_score={result['top_score']:.3f}  "
          f"refused={result['refused']}  docs={len(result['documents'])}")
    return {
        "documents": result["documents"],
        "scores": result["scores"],
        "top_score": result["top_score"],
        "refused": result["refused"],
    }


def generate_node(state: RAGState) -> Dict[str, Any]:
    """Node 3: cited answer generation via Claude."""
    print(f"[generate_node] generating answer over {len(state['documents'])} docs")
    generator = _get_generator()
    retrieval_payload = {
        "documents": state["documents"],
        "refused": False,
    }
    # Use rewritten query for generation so scope clarifications are preserved
    generation_query = state.get("rewritten_query") or state["query"]
    result = generator.generate(generation_query, retrieval_payload)
    print(f"[generate_node] answer length: {len(result['answer'])} chars")
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "refused_reason": None,
    }


def refuse_node(state: RAGState) -> Dict[str, Any]:
    """Node 3: canned refusal when retrieval confidence is too low."""
    print(f"[refuse_node] returning canned refusal "
          f"(top_score={state['top_score']:.3f})")
    return {
        "answer": REFUSAL_RESPONSE,
        "citations": [],
        "refused_reason": "low_retrieval_score",
    }


def route_after_retrieval(state: RAGState) -> str:
    """Conditional edge: refuse if retrieval refused, otherwise generate."""
    return "refuse" if state["refused"] else "generate"


def build_graph():
    """Wire the nodes into a typed StateGraph."""
    workflow = StateGraph(RAGState)

    workflow.add_node("rewrite",  rewrite_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("refuse",   refuse_node)

    workflow.add_edge(START, "rewrite")
    workflow.add_conditional_edges(
        "rewrite",
        route_after_rewrite,
        {"retrieve": "retrieve", "refuse": "refuse"},
    )

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "refuse":   "refuse",
        },
    )

    workflow.add_edge("generate", END)
    workflow.add_edge("refuse",   END)

    return workflow.compile()


def prepare(query: str) -> Dict[str, Any]:
    """
    Run rewrite + retrieve without generation.
    Returns state dict with: query, rewritten_query, documents, scores,
    top_score, refused.  Used by the streaming UI path.
    """
    state: Dict[str, Any] = {"query": query, "refused": False, "top_score": 0.0}

    rewrite_result = rewrite_node(state)
    state.update(rewrite_result)

    if state.get("refused"):
        return state

    retrieve_result = retrieve_node(state)
    state.update(retrieve_result)
    return state


def ask(query: str) -> Dict[str, Any]:
    """Convenience function — run the full RAG pipeline on one question."""
    graph = build_graph()
    final_state = graph.invoke({"query": query})
    return final_state


if __name__ == "__main__":
    test_queries = [
        "Who won the 1998 FIFA World Cup?",
        "What was the Hand of God?",
        "How many goals did Argentina score in 2022?",
        "What is the recipe for chocolate cake?",
    ]

    graph = build_graph()

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        state = graph.invoke({"query": query})

        print(f"\nANSWER:\n{state['answer']}")

        if state.get("citations"):
            print(f"\nCITATIONS:")
            for c in state["citations"]:
                if c["source"] == "wikipedia":
                    print(f"  [{c['index']}] Wikipedia — {c.get('title')}")
                elif c["source"] == "tournament_standings":
                    print(f"  [{c['index']}] Standings — {c.get('team')} ({c.get('year')})")
                elif c["source"] == "tournament_summary":
                    print(f"  [{c['index']}] Tournament summary — {c.get('year')}")
        else:
            print("\n(no citations — refused)")