"""Batch evaluation CLI.

Usage:
    make eval-batch
    python -m evals.run_eval --batch --dataset evals/golden_dataset.json
    python -m evals.run_eval --interactive
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure app is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


async def run_batch(dataset_path: str, limit: int = 0) -> None:
    from app.evaluation.triad import RAGTriadEvaluator
    from app.services.llm import get_llm
    from app.core.config import settings

    dataset: List[Dict[str, Any]] = json.loads(Path(dataset_path).read_text())
    if limit:
        dataset = dataset[:limit]

    evaluator = RAGTriadEvaluator()
    results = []

    console.print(f"\n[bold cyan]Running batch eval on {len(dataset)} samples...[/bold cyan]\n")

    for item in dataset:
        query = item["query"]
        # In real use, you'd call the API here. For now use placeholder context.
        contexts = [f"Sample context for: {query}"]
        answer = f"Sample answer for: {query}"

        cr, gd, ar = await evaluator.evaluate_all(query=query, contexts=contexts, answer=answer)

        passed = (
            cr >= item.get("min_context_relevance", 0.5)
            and gd >= item.get("min_groundedness", 0.5)
            and ar >= item.get("min_answer_relevance", 0.5)
        )
        results.append({
            "id": item["id"],
            "query": query[:50],
            "context_relevance": cr,
            "groundedness": gd,
            "answer_relevance": ar,
            "passed": passed,
        })
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  {item['id']}: {status}  CR={cr:.2f}  GD={gd:.2f}  AR={ar:.2f}")

    # Summary table
    table = Table(title="Batch Eval Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Average", style="magenta")

    avg = lambda k: sum(r[k] for r in results) / len(results)
    table.add_row("Context Relevance", f"{avg('context_relevance'):.3f}")
    table.add_row("Groundedness", f"{avg('groundedness'):.3f}")
    table.add_row("Answer Relevance", f"{avg('answer_relevance'):.3f}")
    passed_count = sum(1 for r in results if r["passed"])
    table.add_row("Pass Rate", f"{passed_count}/{len(results)} ({passed_count/len(results)*100:.1f}%)")

    console.print("\n", table)

    # Save report
    report_path = f"evals/reports/batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(results, indent=2))
    console.print(f"\n[dim]Report saved to {report_path}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--dataset", default="evals/golden_dataset.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.batch:
        asyncio.run(run_batch(args.dataset, args.limit))
    elif args.interactive:
        console.print("[yellow]Interactive eval: enter query and answer to score[/yellow]")
        query = input("Query: ")
        answer = input("Answer: ")
        context = input("Context (or press Enter for empty): ")

        async def _run():
            from app.evaluation.triad import RAGTriadEvaluator
            evaluator = RAGTriadEvaluator()
            cr, gd, ar = await evaluator.evaluate_all(
                query=query, contexts=[context or "no context"], answer=answer
            )
            console.print(f"\nContext Relevance : {cr:.3f}")
            console.print(f"Groundedness      : {gd:.3f}")
            console.print(f"Answer Relevance  : {ar:.3f}")

        asyncio.run(_run())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
