"""
Command-line runner for RAGSS Retrieval Evaluation.

Usage:
  python evaluation/evaluate.py --mode fixture
  python evaluation/evaluate.py --mode live --k 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from evaluation.evaluator import RetrievalEvaluator


def build_fixture_retriever():
    """
    Constructs a deterministic offline mock retriever for fast CI/CD evaluation.
    Matches canonical query keywords against the 3 sample paper chunks without GPU or disk I/O.
    """
    fixture_chunks = {
        "doc_1_paper1_transformers_p0_c0": Document(
            page_content="Title: Attention Is All You Need. Abstract: The dominant sequence transduction models are based on complex recurrence...",
            metadata={"chunk_id": "doc_1_paper1_transformers_p0_c0", "parent_id": "doc_1_paper1_transformers_p0", "document_id": "doc_1_paper1_transformers", "source": "paper1_transformers.pdf", "page": 0},
        ),
        "doc_1_paper1_transformers_p1_c0": Document(
            page_content="Section 2: Scaled Dot-Product Attention. An attention function can be described as mapping a query and a set of key-value pairs...",
            metadata={"chunk_id": "doc_1_paper1_transformers_p1_c0", "parent_id": "doc_1_paper1_transformers_p1", "document_id": "doc_1_paper1_transformers", "source": "paper1_transformers.pdf", "page": 1},
        ),
        "doc_2_paper2_bert_p0_c0": Document(
            page_content="Title: BERT: Pre-training of Deep Bidirectional Transformers. Abstract: We introduce a new language representation model...",
            metadata={"chunk_id": "doc_2_paper2_bert_p0_c0", "parent_id": "doc_2_paper2_bert_p0", "document_id": "doc_2_paper2_bert", "source": "paper2_bert.pdf", "page": 0},
        ),
        "doc_2_paper2_bert_p1_c0": Document(
            page_content="Section 2: Masked Language Model. In order to train a deep bidirectional representation, we simply mask some percentage of input tokens...",
            metadata={"chunk_id": "doc_2_paper2_bert_p1_c0", "parent_id": "doc_2_paper2_bert_p1", "document_id": "doc_2_paper2_bert", "source": "paper2_bert.pdf", "page": 1},
        ),
        "doc_3_paper3_rag_p0_c0": Document(
            page_content="Title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Abstract: Large pre-trained language models explore hybrid retrieval...",
            metadata={"chunk_id": "doc_3_paper3_rag_p0_c0", "parent_id": "doc_3_paper3_rag_p0", "document_id": "doc_3_paper3_rag", "source": "paper3_rag.pdf", "page": 0},
        ),
        "doc_3_paper3_rag_p1_c0": Document(
            page_content="Section 2: Hybrid Retrieval and Generation. RAG combines parametric memory with non-parametric retrieval memory...",
            metadata={"chunk_id": "doc_3_paper3_rag_p1_c0", "parent_id": "doc_3_paper3_rag_p1", "document_id": "doc_3_paper3_rag", "source": "paper3_rag.pdf", "page": 1},
        ),
    }

    def fixture_retrieve(query: str, k: int) -> List[Document]:
        q_lower = query.lower()
        results: List[Document] = []

        # Hand-curated semantic lexical mapping reflecting controlled fixture expectations
        # Use specific phrase 'sequence transduction' to avoid accidental false hits on unanswerable queries
        if "sequence transduction" in q_lower or "recurrence" in q_lower:
            results.append(fixture_chunks["doc_1_paper1_transformers_p0_c0"])
        if "scaled dot-product" in q_lower or ("mapping" in q_lower and "attention" in q_lower):
            results.append(fixture_chunks["doc_1_paper1_transformers_p1_c0"])
        if "deep bidirectional" in q_lower and ("language representation" in q_lower or "pre-training" in q_lower):
            results.append(fixture_chunks["doc_2_paper2_bert_p0_c0"])
        if any(term in q_lower for term in ("mask", "mlm", "masked language model")):
            results.append(fixture_chunks["doc_2_paper2_bert_p1_c0"])
        if "knowledge-intensive" in q_lower or "retrieval-augmented" in q_lower:
            results.append(fixture_chunks["doc_3_paper3_rag_p0_c0"])
        if "parametric" in q_lower or "non-parametric" in q_lower:
            results.append(fixture_chunks["doc_3_paper3_rag_p1_c0"])
        if "contrast" in q_lower or ("papers" in q_lower and "attention" in q_lower and "masked" in q_lower):
            results.extend([
                fixture_chunks["doc_1_paper1_transformers_p1_c0"],
                fixture_chunks["doc_2_paper2_bert_p1_c0"],
            ])
        if "compare" in q_lower and "transformer" in q_lower and "retrieval-augmented" in q_lower:
            results.extend([
                fixture_chunks["doc_1_paper1_transformers_p0_c0"],
                fixture_chunks["doc_3_paper3_rag_p0_c0"],
            ])

        # Deduplicate preserving order
        deduped: List[Document] = []
        seen = set()
        for doc in results:
            cid = doc.metadata.get("chunk_id")
            if cid not in seen:
                seen.add(cid)
                deduped.append(doc)

        return deduped[:k]

    return fixture_retrieve


def build_live_retriever(k: int = 5):
    """Initializes live Retriever against the sample papers corpus."""
    from retrieval.retriver import Retriever

    sample_dir = PROJECT_ROOT / "tests" / "sample_papers"
    sample_files = sorted(list(sample_dir.glob("*.pdf")))
    if not sample_files:
        raise FileNotFoundError(f"No sample papers found in {sample_dir}")

    eval_db_dir = PROJECT_ROOT / "tests" / "test_eval_db"
    retriever = Retriever(
        file_paths=[str(f) for f in sample_files],
        persist_directory=str(eval_db_dir),
        collection_name="eval_sample_papers",
        rebuild_index=True,
    )

    def live_retrieve(query: str, top_k: int) -> List[Document]:
        return retriever.retrieve(query=query, k=top_k)

    return live_retrieve


def run_evaluation(
    dataset_path: str,
    mode: str = "fixture",
    target_id_level: str = "parent",
    max_k: int = 5,
    output_json_path: Optional[str] = None,
    quiet: bool = False,
):
    if not quiet:
        print("=" * 70)
        print(" RAGSS RETRIEVAL EVALUATION HARNESS")
        print("=" * 70)
        print(f"Evaluation Mode: {mode.upper()}")
        print(f"Dataset Path:    {dataset_path}")
        print(f"Target ID Level: {target_id_level}")
        print(f"Max Retrieval K: {max_k}\n")

    evaluator = RetrievalEvaluator(dataset_path=dataset_path, target_id_level=target_id_level)

    if mode == "fixture":
        if not quiet:
            print("Initializing deterministic fixture retriever...")
        retriever_fn = build_fixture_retriever()
    else:
        if not quiet:
            print("Initializing live Retriever against sample papers...")
        retriever_fn = build_live_retriever(k=max_k)

    if not quiet:
        print(f"Running evaluation over {len(evaluator.dataset)} queries...\n")
    metrics = evaluator.evaluate(retriever_fn=retriever_fn, max_k=max_k)

    if not quiet:
        print("=" * 70)
        print("PER-QUERY RESULTS")
        print("=" * 70)
        for res in metrics.per_query_results:
            type_str = "Answerable" if res.is_answerable else "Unanswerable"
            if res.is_answerable:
                status_icon = "✓" if res.has_any_hit else "✗"
                print(f"{status_icon} [{res.query_id}] ({type_str}) - {res.question[:55]}...")
                print(f"    Expected:  {res.expected_ids}")
                print(f"    Retrieved: {res.retrieved_ids}")
                print(f"    R@1: {res.recall_at_1:.2f} | R@3: {res.recall_at_3:.2f} | R@5: {res.recall_at_5:.2f} | P@5: {res.precision_at_5:.2f} | MRR: {res.mrr:.2f} | Latency: {res.latency_ms:.2f}ms")
            else:
                if res.empty_retrieval:
                    status_icon = "✓"
                    print(f"{status_icon} [{res.query_id}] ({type_str}) - {res.question[:55]}...")
                    print("    Out-of-corpus query | Clean abstention (0 docs retrieved)")
                else:
                    status_icon = "✗"
                    print(f"{status_icon} [{res.query_id}] ({type_str}) - {res.question[:55]}...")
                    print(f"    Out-of-corpus query | False positive retrieval ({len(res.retrieved_ids)} doc(s) retrieved: {res.retrieved_ids})")
            print("-" * 70)

        print("\n" + "=" * 70)
        print("AGGREGATE BENCHMARK RESULTS")
        print("=" * 70)
        print(metrics.summary_table())

    if output_json_path:
        out_path = Path(output_json_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_payload = metrics.to_dict()
        summary_payload["mode"] = mode
        summary_payload["target_id_level"] = target_id_level
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        if not quiet:
            print(f"✓ Saved detailed results JSON to: {out_path}")

    return metrics


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Evaluate RAGSS retrieval performance.")
    parser.add_argument(
        "--mode",
        choices=["fixture", "live"],
        default="fixture",
        help="Evaluation mode: 'fixture' (fast, offline, deterministic) or 'live' (real embeddings + BM25 index).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "retrieval_dataset.jsonl"),
        help="Path to evaluation dataset JSONL.",
    )
    parser.add_argument(
        "--level",
        choices=["chunk", "parent", "document"],
        default="parent",
        help="Target ID evaluation level (chunk, parent, or document). Defaults to 'parent' matching Retriever.retrieve expansion.",
    )
    parser.add_argument("--k", type=int, default=5, help="Maximum K for retrieval.")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write full evaluation results as JSON.",
    )
    args = parser.parse_args(argv)

    run_evaluation(
        dataset_path=args.dataset,
        mode=args.mode,
        target_id_level=args.level,
        max_k=args.k,
        output_json_path=args.output_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
