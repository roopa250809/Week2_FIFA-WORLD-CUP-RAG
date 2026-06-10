from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import ANTHROPIC_API_KEY, LLM_MODEL


SYSTEM_PROMPT = """You are a FIFA World Cup expert assistant. You answer questions \
strictly from the provided context, which contains tournament standings, summaries, \
and Wikipedia articles about World Cup history, players, and iconic moments.

Strict rules:
1. Use ONLY the information in the numbered context sources below. Do not use any \
outside knowledge, even if you think you know the answer.
2. If the context does not contain enough information to answer, say so honestly: \
"I couldn't find enough information in the World Cup records to answer that."
3. Cite your sources inline using bracket notation like [1], [2], [3] referring to \
the source numbers. Cite every factual claim.
4. Be concise. Answer in 2-4 sentences unless the question genuinely requires more.
5. If sources disagree or are ambiguous, acknowledge that explicitly.
6. Begin your answer by directly addressing the question using its own key terms. \
For example: "What was the Hand of God?" → start "The Hand of God was..."; \
"Tell me about the Maracanazo" → start "The Maracanazo was..."; \
"Who won the 1998 World Cup?" → start "France won the 1998 FIFA World Cup...".

Important clarifications to avoid common mistakes:
- "TEAMS" in tournament data refers to the NUMBER OF TEAMS that participated, NOT the \
edition number of the tournament. For example "15 teams competed" means 15 countries \
participated, it does NOT mean this was the 15th World Cup.
- The World Cup edition order by year: 1930=1st, 1934=2nd, 1938=3rd, 1950=4th, \
1954=5th, 1958=6th, 1962=7th, 1966=8th, 1970=9th, 1974=10th, 1978=11th, 1982=12th, \
1986=13th, 1990=14th, 1994=15th, 1998=16th, 2002=17th, 2006=18th, 2010=19th, \
2014=20th, 2018=21st, 2022=22nd.
- If the user asks about "the 3rd World Cup" and the context contains 1938 data, \
answer from that data — that is the correct match.
- When asked how many goals a player scored "in [year]" or "in the [year] FIFA World Cup", \
answer with the World Cup figure directly and concisely. Do NOT offer alternative \
interpretations or mention other competitions — give the World Cup answer and stop.

Never fabricate scores, dates, player names, or statistics. If you're uncertain, say so."""


REFUSAL_RESPONSE = (
    "I couldn't find that information in the World Cup records. "
    "I can answer questions about tournament results, team performances, "
    "famous players, and iconic World Cup moments from 1930 to 2022. "
    "Try rephrasing your question or asking about a specific tournament or player."
)


class AnswerGenerator:
    """Generate cited answers from retrieved context using Claude."""

    def __init__(self):
        print(f"Initializing LLM: {LLM_MODEL}")
        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0.0,
            max_tokens=600,
        )

    @staticmethod
    def _format_context(documents: List[Document]) -> str:
        """Format retrieved chunks as numbered sources for the LLM."""
        blocks = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            if source == "wikipedia":
                label = f"Wikipedia: {doc.metadata.get('title', 'untitled')}"
            elif source == "tournament_standings":
                label = f"Standings: {doc.metadata.get('team')} in {doc.metadata.get('year')}"
            elif source == "tournament_summary":
                label = f"Tournament summary: {doc.metadata.get('year')}"
            else:
                label = source

            blocks.append(f"[{i}] {label}\n{doc.page_content}")

        return "\n\n---\n\n".join(blocks)

    @staticmethod
    def _format_citations(documents: List[Document]) -> List[Dict[str, Any]]:
        """Build a clean citations list to return alongside the answer."""
        citations = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            entry = {"index": i, "source": source}
            if source == "wikipedia":
                entry["title"] = doc.metadata.get("title")
                entry["url"]   = doc.metadata.get("url")
            elif source == "tournament_standings":
                entry["team"] = doc.metadata.get("team")
                entry["year"] = doc.metadata.get("year")
            elif source == "tournament_summary":
                entry["year"] = doc.metadata.get("year")
            citations.append(entry)
        return citations

    def stream_generate(self, query: str, retrieval_result: Dict[str, Any]):
        """
        Streaming variant — returns (chunk_generator, citations, refused).
        Use with st.write_stream(generator).
        """
        if retrieval_result.get("refused"):
            def _refusal():
                yield REFUSAL_RESPONSE
            return _refusal(), [], True

        documents = retrieval_result["documents"]
        context   = self._format_context(documents)
        citations = self._format_citations(documents)

        user_prompt = (
            f"Context sources:\n\n{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely using only the sources above. "
            f"Cite each claim with [1], [2], etc."
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        def _stream():
            for chunk in self.llm.stream(messages):
                yield chunk.content

        return _stream(), citations, False

    def generate(self, query: str, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Take a retrieval result and produce an answer.
        Returns: {answer, citations, refused, refused_reason}
        """
        if retrieval_result["refused"]:
            return {
                "answer": REFUSAL_RESPONSE,
                "citations": [],
                "refused": True,
                "refused_reason": "low_retrieval_score",
            }

        documents = retrieval_result["documents"]
        context = self._format_context(documents)

        user_prompt = (
            f"Context sources:\n\n{context}\n\n"
            f"---\n\n"
            f"Question: {query}\n\n"
            f"Answer concisely using only the sources above. "
            f"Cite each claim with [1], [2], etc."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        answer = response.content.strip()

        return {
            "answer": answer,
            "citations": self._format_citations(documents),
            "refused": False,
            "refused_reason": None,
        }


if __name__ == "__main__":
    from src.retrieve import HybridRetriever

    retriever = HybridRetriever()
    generator = AnswerGenerator()

    test_queries = [
        "Who won the 1998 FIFA World Cup?",
        "What was the Hand of God?",
        "How many goals did Argentina score in 2022?",
        "What is the recipe for chocolate cake?",
    ]

    for query in test_queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        retrieval = retriever.retrieve(query)
        print(f"Retrieval — top dense: {retrieval['top_score']:.3f}  "
              f"refused: {retrieval['refused']}")

        result = generator.generate(query, retrieval)

        print(f"\nANSWER:\n{result['answer']}")

        if result["citations"]:
            print(f"\nCITATIONS:")
            for c in result["citations"]:
                if c["source"] == "wikipedia":
                    print(f"  [{c['index']}] Wikipedia — {c.get('title')}")
                elif c["source"] == "tournament_standings":
                    print(f"  [{c['index']}] Standings — {c.get('team')} ({c.get('year')})")
                elif c["source"] == "tournament_summary":
                    print(f"  [{c['index']}] Tournament summary — {c.get('year')}")