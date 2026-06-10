"""
Golden dataset evaluation for the FIFA World Cup RAG pipeline.

Run:
    python -m tests.test_rag
    python -m tests.test_rag --category ordinal_rewrite
    python -m tests.test_rag --verbose
"""

import argparse
import time
import pandas as pd
from src.graph import build_graph

DATASET_PATH = "tests/golden_dataset.csv"

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def evaluate(category_filter: str = None, verbose: bool = False):
    df = pd.read_csv(DATASET_PATH)
    if category_filter:
        df = df[df["category"] == category_filter]
        if df.empty:
            print(f"No rows found for category '{category_filter}'")
            return

    graph = build_graph()

    results = []
    print(f"\n{BOLD}Running {len(df)} test cases...{RESET}\n")

    for _, row in df.iterrows():
        question        = row["question"]
        expected        = str(row["expected_contains"]).strip()
        should_refuse   = bool(row["should_refuse"])
        category        = row["category"]
        notes           = row.get("notes", "")

        t0 = time.time()
        state = graph.invoke({"query": question})
        latency = round(time.time() - t0, 1)

        refused  = state.get("refused", False)
        answer   = state.get("answer", "")
        rewritten = state.get("rewritten_query", question)
        score    = state.get("top_score", 0.0)

        if should_refuse:
            passed = refused
        else:
            passed = not refused and expected.lower() in answer.lower()

        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"

        print(f"{status}  [{category}]  {question}")
        if not passed:
            if should_refuse and not refused:
                print(f"       {RED}Expected refusal but got answer:{RESET} {answer[:120]}")
            elif not should_refuse and refused:
                print(f"       {RED}Unexpectedly refused (score={score:.3f}){RESET}")
            elif not should_refuse and expected.lower() not in answer.lower():
                print(f"       {RED}Expected '{expected}' not found in answer:{RESET}")
                print(f"       {answer[:120]}")

        if verbose:
            if rewritten != question:
                print(f"       {CYAN}Rewritten:{RESET} {rewritten}")
            print(f"       {YELLOW}Score:{RESET} {score:.3f}  {YELLOW}Latency:{RESET} {latency}s")
            print(f"       {YELLOW}Answer:{RESET} {answer[:150]}")

        results.append({
            "question":  question,
            "category":  category,
            "passed":    passed,
            "latency":   latency,
            "score":     score,
            "refused":   refused,
            "notes":     notes,
        })

    # Summary
    total   = len(results)
    passed  = sum(r["passed"] for r in results)
    failed  = total - passed
    avg_lat = round(sum(r["latency"] for r in results) / total, 1)

    print(f"\n{BOLD}{'='*55}{RESET}")
    print(f"{BOLD}Results: {GREEN}{passed} passed{RESET}{BOLD}, {RED}{failed} failed{RESET}{BOLD} / {total} total{RESET}")
    print(f"{BOLD}Accuracy: {round(passed/total*100, 1)}%{RESET}")
    print(f"{BOLD}Avg latency: {avg_lat}s{RESET}")

    # Per-category breakdown
    df_results = pd.DataFrame(results)
    print(f"\n{BOLD}By category:{RESET}")
    for cat, group in df_results.groupby("category"):
        cat_pass = group["passed"].sum()
        cat_total = len(group)
        bar = f"{GREEN}{'|'*cat_pass}{RED}{'|'*(cat_total-cat_pass)}{RESET}"
        print(f"  {cat:<25} {cat_pass}/{cat_total}  {bar}")

    # Flag slow queries
    slow = [r for r in results if r["latency"] > 8.0]
    if slow:
        print(f"\n{YELLOW}Slow queries (>8s):{RESET}")
        for r in slow:
            print(f"  {r['latency']}s  {r['question']}")

    print()
    return passed, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG golden dataset evaluator")
    parser.add_argument("--category", help="Filter by category", default=None)
    parser.add_argument("--verbose",  help="Show answer + latency per test", action="store_true")
    args = parser.parse_args()
    evaluate(category_filter=args.category, verbose=args.verbose)
