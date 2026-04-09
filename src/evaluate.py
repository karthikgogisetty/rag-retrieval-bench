"""
Main evaluation orchestrator.

Runs the full pipeline:
1. Chunk the document
2. Build ground truth
3. For each embedding model:
   a. Embed all chunks
   b. Embed all queries
   c. Retrieve top-K for each query
   d. Compute metrics
4. Aggregate and compare results
"""

import json
import os
import time
from dataclasses import asdict

from .chunker import chunk_document, Chunk
from .ground_truth import build_ground_truth, EvalQuestion
from .embeddings import EmbeddingEngine, MODEL_CONFIGS
from .retrieval import retrieve_top_k, RetrievalResult
from .metrics import compute_query_metrics, aggregate_metrics, QueryMetrics


def format_table(headers: list[str], rows: list[list], col_width: int = 18) -> str:
    """Format a simple text table."""
    header_line = " | ".join(h.ljust(col_width) for h in headers)
    separator = "-+-".join("-" * col_width for _ in headers)
    lines = [header_line, separator]
    for row in rows:
        line = " | ".join(str(v).ljust(col_width) for v in row)
        lines.append(line)
    return "\n".join(lines)


def run_evaluation(doc_path: str, results_dir: str, max_k: int = 5) -> dict:
    """
    Run the complete evaluation pipeline.

    Returns a dict with all results for serialization.
    """
    print("=" * 70)
    print("EMBEDDING MODEL EVALUATION SYSTEM")
    print("=" * 70)

    # Step 1: Chunk the document
    print("\n[1/4] Chunking document...")
    chunks = chunk_document(doc_path)
    print(f"  Created {len(chunks)} chunks")
    print(f"  Token range: {min(c.token_count for c in chunks)}-{max(c.token_count for c in chunks)}")
    print(f"  Avg tokens: {sum(c.token_count for c in chunks) // len(chunks)}")

    # Step 2: Build ground truth
    print("\n[2/4] Building ground truth annotations...")
    questions = build_ground_truth(chunks)
    for q in questions:
        print(f"  Q{q.qid} ({q.description}): {len(q.relevant_chunks)} relevant chunks "
              f"(grades: {dict(sorted(q.relevant_chunks.items()))})")

    # Prepare chunk data for embedding
    chunk_texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    query_texts = [q.question for q in questions]

    # Step 3: Evaluate each model
    print("\n[3/4] Evaluating embedding models...")
    all_results = {}

    for config in MODEL_CONFIGS:
        print(f"\n{'─' * 50}")
        print(f"Model: {config.name}")
        print(f"  {config.description}")
        print(f"{'─' * 50}")

        # Load model
        t0 = time.time()
        engine = EmbeddingEngine(config)
        load_time = time.time() - t0

        # Embed passages
        t0 = time.time()
        passage_embeddings = engine.embed_passages(chunk_texts)
        embed_passage_time = time.time() - t0
        print(f"  Embedded {len(chunk_texts)} chunks in {embed_passage_time:.2f}s")

        # Embed queries
        t0 = time.time()
        query_embeddings = engine.embed_queries(query_texts)
        embed_query_time = time.time() - t0
        print(f"  Embedded {len(query_texts)} queries in {embed_query_time:.2f}s")

        # Retrieve top-K
        retrieval_results = retrieve_top_k(
            query_embeddings, passage_embeddings, chunk_ids, k=max_k
        )

        # Compute per-query metrics
        per_query_metrics = []
        per_query_details = []

        for q_idx, (question, results) in enumerate(zip(questions, retrieval_results)):
            retrieved_ids = [r.chunk_id for r in results]
            qm = compute_query_metrics(question.qid, retrieved_ids, question.relevant_chunks)
            per_query_metrics.append(qm)

            # Store detailed results for reporting
            detail = {
                "qid": question.qid,
                "description": question.description,
                "retrieved_chunks": [],
            }
            for r in results:
                chunk = next(c for c in chunks if c.chunk_id == r.chunk_id)
                is_relevant = r.chunk_id in question.relevant_chunks
                grade = question.relevant_chunks.get(r.chunk_id, 0)
                detail["retrieved_chunks"].append({
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "score": round(r.score, 4),
                    "section_id": chunk.section_id,
                    "section_title": chunk.section_title,
                    "relevant": is_relevant,
                    "grade": grade,
                    "text_preview": chunk.text[:120],
                })
            per_query_details.append(detail)

        # Aggregate metrics
        agg = aggregate_metrics(per_query_metrics)

        model_result = {
            "model_name": config.name,
            "model_id": config.model_id,
            "description": config.description,
            "dim": config.dim,
            "load_time_s": round(load_time, 2),
            "embed_passage_time_s": round(embed_passage_time, 2),
            "embed_query_time_s": round(embed_query_time, 2),
            "aggregate_metrics": {k: round(v, 4) for k, v in agg.items()},
            "per_query_metrics": [asdict(qm) for qm in per_query_metrics],
            "per_query_details": per_query_details,
        }
        all_results[config.name] = model_result

        # Print summary
        print(f"\n  Aggregate Metrics:")
        for metric_name, value in agg.items():
            print(f"    {metric_name:<20} {value:.4f}")

    # Step 4: Compare and rank
    print("\n" + "=" * 70)
    print("[4/4] COMPARATIVE RESULTS")
    print("=" * 70)

    # Build comparison table
    metric_names = [
        "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
        "precision_at_3", "precision_at_5",
        "recall_at_3", "recall_at_5",
        "mrr", "ndcg_at_3", "ndcg_at_5",
    ]

    headers = ["Metric"] + [config.name for config in MODEL_CONFIGS]
    rows = []
    for mname in metric_names:
        row = [mname]
        values = []
        for config in MODEL_CONFIGS:
            val = all_results[config.name]["aggregate_metrics"][mname]
            values.append(val)
            row.append(f"{val:.4f}")
        # Mark the winner with an asterisk
        max_val = max(values)
        for i, v in enumerate(values):
            if v == max_val:
                row[i + 1] = f"{v:.4f} *"
        rows.append(row)

    print("\n" + format_table(headers, rows, col_width=20))
    print("\n  (* = best for that metric)")

    # Determine overall winner by NDCG@5 (primary metric)
    print("\n" + "─" * 70)
    print("RANKING (by NDCG@5, primary metric):")
    ranked = sorted(
        MODEL_CONFIGS,
        key=lambda c: all_results[c.name]["aggregate_metrics"]["ndcg_at_5"],
        reverse=True,
    )
    for i, config in enumerate(ranked, 1):
        ndcg5 = all_results[config.name]["aggregate_metrics"]["ndcg_at_5"]
        mrr = all_results[config.name]["aggregate_metrics"]["mrr"]
        recall5 = all_results[config.name]["aggregate_metrics"]["recall_at_5"]
        print(f"  #{i}: {config.name:<25} NDCG@5={ndcg5:.4f}  MRR={mrr:.4f}  Recall@5={recall5:.4f}")

    # Per-query breakdown
    print("\n" + "─" * 70)
    print("PER-QUERY NDCG@5 BREAKDOWN:")
    q_headers = ["Question"] + [c.name for c in MODEL_CONFIGS]
    q_rows = []
    for q in questions:
        row = [f"Q{q.qid}: {q.description[:25]}"]
        for config in MODEL_CONFIGS:
            qm_list = all_results[config.name]["per_query_metrics"]
            qm = next(m for m in qm_list if m["qid"] == q.qid)
            row.append(f"{qm['ndcg_at_5']:.4f}")
        q_rows.append(row)
    print("\n" + format_table(q_headers, q_rows, col_width=25))

    # Per-query retrieved chunks detail
    print("\n" + "─" * 70)
    print("DETAILED RETRIEVAL RESULTS (Top-5 per query per model):")
    for q in questions:
        print(f"\n  Q{q.qid}: {q.description}")
        print(f"  Question: {q.question[:100]}...")
        print(f"  Ground truth chunks: {dict(sorted(q.relevant_chunks.items()))}")
        for config in MODEL_CONFIGS:
            detail = next(
                d for d in all_results[config.name]["per_query_details"]
                if d["qid"] == q.qid
            )
            print(f"\n    [{config.name}]")
            for rc in detail["retrieved_chunks"]:
                marker = "HIT" if rc["relevant"] else "   "
                grade_str = f"g={rc['grade']}" if rc["relevant"] else "    "
                print(f"      #{rc['rank']} [{marker}] {grade_str} "
                      f"score={rc['score']:.4f} "
                      f"chunk={rc['chunk_id']:02d} sec={rc['section_id']:<5} "
                      f"| {rc['text_preview'][:70]}...")

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(results_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nResults saved to {json_path}")

    # Generate and save text report
    report = _generate_report(all_results, chunks, questions, MODEL_CONFIGS)
    report_path = os.path.join(results_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    return all_results


def _generate_report(
    all_results: dict,
    chunks: list[Chunk],
    questions: list[EvalQuestion],
    model_configs: list,
) -> str:
    """Generate a comprehensive text report."""
    lines = []
    lines.append("=" * 70)
    lines.append("EMBEDDING MODEL EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # 1. Setup summary
    lines.append("1. EVALUATION SETUP")
    lines.append(f"   Document: Acme Enterprise Platform (22 pages)")
    lines.append(f"   Chunks: {len(chunks)}")
    lines.append(f"   Queries: {len(questions)}")
    lines.append(f"   Models evaluated: {len(model_configs)}")
    lines.append("")

    # 2. Models
    lines.append("2. MODELS EVALUATED")
    for config in model_configs:
        lines.append(f"   - {config.name}: {config.description}")
    lines.append("")

    # 3. Ground truth summary
    lines.append("3. GROUND TRUTH SUMMARY")
    for q in questions:
        high = sum(1 for g in q.relevant_chunks.values() if g == 2)
        partial = sum(1 for g in q.relevant_chunks.values() if g == 1)
        lines.append(f"   Q{q.qid} ({q.description}): "
                      f"{high} highly relevant + {partial} partially relevant chunks")
    lines.append("")

    # 4. Aggregate results table
    lines.append("4. AGGREGATE RESULTS (macro-averaged across 5 queries)")
    lines.append("")
    metric_names = [
        "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5",
        "precision_at_3", "precision_at_5",
        "recall_at_3", "recall_at_5",
        "mrr", "ndcg_at_3", "ndcg_at_5",
    ]
    header = f"   {'Metric':<20}"
    for config in model_configs:
        header += f" {config.name:>20}"
    lines.append(header)
    lines.append("   " + "-" * (20 + 21 * len(model_configs)))
    for mname in metric_names:
        row = f"   {mname:<20}"
        values = []
        for config in model_configs:
            val = all_results[config.name]["aggregate_metrics"][mname]
            values.append(val)
        max_val = max(values)
        for val in values:
            marker = " *" if val == max_val else "  "
            row += f" {val:>18.4f}{marker}"
        lines.append(row)
    lines.append("")
    lines.append("   (* = best)")
    lines.append("")

    # 5. Winner
    ranked = sorted(
        model_configs,
        key=lambda c: all_results[c.name]["aggregate_metrics"]["ndcg_at_5"],
        reverse=True,
    )
    winner = ranked[0]
    ndcg5 = all_results[winner.name]["aggregate_metrics"]["ndcg_at_5"]
    lines.append("5. CONCLUSION")
    lines.append(f"   Winner: {winner.name} (NDCG@5 = {ndcg5:.4f})")
    lines.append("")
    lines.append(f"   Ranking by NDCG@5 (primary metric):")
    for i, config in enumerate(ranked, 1):
        n5 = all_results[config.name]["aggregate_metrics"]["ndcg_at_5"]
        lines.append(f"     #{i}: {config.name} (NDCG@5={n5:.4f})")
    lines.append("")

    # 6. Per-query breakdown
    lines.append("6. PER-QUERY NDCG@5")
    for q in questions:
        lines.append(f"\n   Q{q.qid}: {q.description}")
        for config in model_configs:
            qm_list = all_results[config.name]["per_query_metrics"]
            qm = next(m for m in qm_list if m["qid"] == q.qid)
            lines.append(f"     {config.name:<25} NDCG@5={qm['ndcg_at_5']:.4f}  "
                          f"MRR={qm['mrr']:.4f}  Recall@5={qm['recall_at_5']:.4f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)
