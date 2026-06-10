"""
RAGAS faithfulness evaluation for the FIFA World Cup RAG pipeline.

Uses the official RAGAS library (0.3.x) with Claude as the judge LLM.

Metrics:
  - Faithfulness:      fraction of answer claims supported by retrieved contexts
  - AnswerRelevancy:   how well the answer addresses the question
  - ContextRecall:     fraction of ground-truth info present in retrieved contexts

Run:
    python -m tests.ragas_eval
    python -m tests.ragas_eval --verbose
    python -m tests.ragas_eval --max-rows 5
"""

import argparse
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, EMBED_MODEL
from src.graph import build_graph

DATASET_PATH = "tests/golden_dataset.csv"

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def build_ragas_llm() -> LangchainLLMWrapper:
    llm = ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0.0,
        max_tokens=2048,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return LangchainEmbeddingsWrapper(emb)


def evaluate_pipeline(max_rows: int | None = None, verbose: bool = False):
    df = pd.read_csv(DATASET_PATH)

    # Skip out-of-domain/refused rows — faithfulness is undefined for refusals
    eval_df = df[df["should_refuse"].astype(str).str.lower() != "true"].copy()
    if max_rows:
        eval_df = eval_df.head(max_rows)

    print(f"\n{BOLD}Running RAG pipeline on {len(eval_df)} questions...{RESET}\n")

    graph = build_graph()
    samples = []

    for _, row in eval_df.iterrows():
        question     = row["question"]
        ground_truth = str(row["expected_contains"])

        t0 = time.time()
        state = graph.invoke({"query": question})
        latency = round(time.time() - t0, 1)

        answer   = state.get("answer", "")
        docs     = state.get("documents", [])
        contexts = [d.page_content for d in docs] if docs else []

        print(f"  [{latency}s] {question[:70]}")
        if verbose:
            print(f"    Answer: {answer[:120]}")

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    dataset = EvaluationDataset(samples=samples)

    print(f"\n{BOLD}Running RAGAS evaluation (judge: {LLM_MODEL})...{RESET}\n")

    ragas_llm = build_ragas_llm()
    ragas_emb = build_ragas_embeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        ContextRecall(llm=ragas_llm),
    ]

    results = evaluate(dataset=dataset, metrics=metrics)

    # Print per-question scores
    scores_df = results.to_pandas()
    print(f"{'Question':<55} {'Faith':>6} {'Relev':>6} {'Recall':>7}")
    print("-" * 78)

    for _, r in scores_df.iterrows():
        q = str(r["user_input"])
        q_display = q[:52] + "..." if len(q) > 52 else q
        f  = float(r.get("faithfulness", 0) or 0)
        rv = float(r.get("answer_relevancy", 0) or 0)
        rc = float(r.get("context_recall", 0) or 0)

        f_col  = GREEN if f  >= 0.80 else YELLOW
        rv_col = GREEN if rv >= 0.80 else YELLOW
        rc_col = GREEN if rc >= 0.80 else YELLOW

        print(
            f"{q_display:<55} "
            f"{f_col}{f:>6.2f}{RESET} "
            f"{rv_col}{rv:>6.2f}{RESET} "
            f"{rc_col}{rc:>7.2f}{RESET}"
        )

    # Aggregate scores
    avg_f  = scores_df["faithfulness"].mean()
    avg_r  = scores_df["answer_relevancy"].mean()
    avg_rc = scores_df["context_recall"].mean()

    target_f  = 0.90
    target_r  = 0.85
    target_rc = 0.80

    def _check(val, target):
        return f"{GREEN}PASS{RESET}" if val >= target else f"{RED}FAIL{RESET}"

    print("\n" + "=" * 78)
    print(f"{BOLD}RAGAS Scores (n={len(samples)} questions, judge: Claude){RESET}")
    print(f"  Faithfulness      {avg_f:.4f}  target {target_f}   {_check(avg_f,  target_f)}")
    print(f"  Answer Relevancy  {avg_r:.4f}  target {target_r}  {_check(avg_r,  target_r)}")
    print(f"  Context Recall    {avg_rc:.4f}  target {target_rc}   {_check(avg_rc, target_rc)}")

    # Per-category breakdown
    scores_df["category"] = eval_df["category"].values
    print(f"\n{BOLD}By category:{RESET}")
    for cat, grp in scores_df.groupby("category"):
        cf  = grp["faithfulness"].mean()
        cr  = grp["answer_relevancy"].mean()
        print(f"  {cat:<25}  faith={cf:.2f}  relevancy={cr:.2f}  (n={len(grp)})")

    print()
    return avg_f, avg_r, avg_rc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS evaluation for FIFA World Cup RAG")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Limit to first N answerable questions (default: all 14)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print answer text for each question")
    args = parser.parse_args()
    evaluate_pipeline(max_rows=args.max_rows, verbose=args.verbose)
