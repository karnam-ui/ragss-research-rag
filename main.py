#!/usr/bin/env python3
"""
RAGSS Unified Command-Line Interface (CLI)

Provides a unified interface for:
  - query: Grounded question answering with evidence citations
  - analyze: Structured paper-level analysis (single or multi-paper)
  - compare: Multi-paper comparative research synthesis
  - evaluate: Offline or live retrieval benchmark harness
  - ingest: Hierarchical document ingestion and vector indexing
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.config import RAGConfig
from retrieval.evidence import Evidence, build_citations

logger = logging.getLogger("ragss")


def _emit_error(args: Optional[argparse.Namespace], message: str, code: int = 1) -> int:
    """
    Emits error diagnostics to sys.stderr, and if JSON output was requested,
    writes a structured JSON error object to sys.stdout so stdout remains parseable.
    """
    print(f"❌ Error: {message}", file=sys.stderr)
    if args is not None and getattr(args, "json", False):
        payload = {
            "status": "error",
            "error": message,
            "exit_code": code,
        }
        print(json.dumps(payload, indent=2), file=sys.stdout)
    return code


def _emit_internal_error(args: Optional[argparse.Namespace], message: str, code: int = 2) -> int:
    """
    Emits unexpected internal error diagnostics to sys.stderr, and structured JSON to stdout.
    """
    print(f"❌ Internal Error: {message}", file=sys.stderr)
    if args is not None and getattr(args, "json", False):
        payload = {
            "status": "internal_error",
            "error": message,
            "exit_code": code,
        }
        print(json.dumps(payload, indent=2), file=sys.stdout)
    return code


def _resolve_pdf_paths(paths_or_dirs: List[str]) -> List[str]:
    """Resolves individual files or directories into a sorted list of PDF paths."""
    resolved: List[Path] = []
    for p_str in paths_or_dirs:
        p = Path(p_str).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")
        if p.is_dir():
            pdfs = sorted(list(p.glob("*.pdf")))
            if not pdfs:
                raise ValueError(f"No PDF files found in directory: {p}")
            resolved.extend(pdfs)
        elif p.suffix.lower() == ".pdf":
            resolved.append(p)
        else:
            raise ValueError(f"File is not a PDF: {p}")

    deduped = sorted(list({str(p) for p in resolved}))
    if not deduped:
        raise ValueError("No valid PDF files provided.")
    return deduped


# ----------------------------------------------------------------------
# Command Handlers
# ----------------------------------------------------------------------

def cmd_query(args: argparse.Namespace) -> int:
    """Executes grounded question answering with citations."""
    from retrieval.pipeline import BasicRetriever
    from retrieval.retriver import Retriever

    query_text = args.query.strip()
    if not query_text:
        return _emit_error(args, "Query text cannot be empty.")

    if args.k <= 0:
        return _emit_error(args, f"Retrieval K must be a positive integer, got {args.k}.")

    file_paths = None
    if args.docs:
        try:
            file_paths = _resolve_pdf_paths(args.docs)
        except (FileNotFoundError, ValueError) as err:
            return _emit_error(args, str(err))

    cfg = RAGConfig()
    retriever = Retriever(
        file_paths=file_paths,
        persist_directory=args.db_dir or cfg.chroma_dir,
    )

    if args.no_llm:
        # Retrieval preview: runs embedding, BM25, RRF, cross-encoder, and parent expansion
        # but skips LLM token generation and does not instantiate the LLM chain.
        results = retriever.retrieve(query=query_text, k=args.k)
        citations = []
        for idx, doc in enumerate(results, 1):
            raw_page = doc.metadata.get("page", 0)
            try:
                page_num = int(raw_page) + 1
            except (ValueError, TypeError):
                page_num = 1
            citations.append({
                "rank": idx,
                "document_id": doc.metadata.get("document_id"),
                "document_name": doc.metadata.get("document_name"),
                "page": page_num,
                "snippet": doc.page_content[:200].replace("\n", " ") + "...",
            })

        output_payload = {
            "query": query_text,
            "mode": "retrieval_only",
            "k": args.k,
            "results_count": len(results),
            "citations": citations,
        }

        if args.json:
            print(json.dumps(output_payload, indent=2))
        else:
            print(f"\nQUERY: {query_text}")
            print(f"RETRIEVED CHUNKS: {len(results)}\n" + "-" * 60)
            for item in citations:
                print(f"[{item['rank']}] {item['document_name']} (p.{item['page']}): {item['snippet']}")
        return 0

    pipeline = BasicRetriever(retriever=retriever)
    output = pipeline.generate(query_text)

    if args.json:
        payload = {
            "query": query_text,
            "answer": output.get("answer", "").strip(),
            "citations": output.get("citations", []),
        }
        print(json.dumps(payload, indent=2))
    else:
        print("\n" + "=" * 70)
        print(f"QUESTION: {query_text}")
        print("=" * 70)
        print("ANSWER:\n" + output.get("answer", "").strip())
        print("\nCITATIONS:")
        citations = output.get("citations", [])
        if citations:
            for cit in citations:
                print(f"  - [{cit.get('document_name')}] Page {cit.get('page')}: {cit.get('source')}")
        else:
            print("  (None)")
        print("=" * 70)

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Executes structured single or multi-paper analysis."""
    from retrieval.paper_analysis import analyze_papers
    from retrieval.retriver import Retriever

    if args.k <= 0:
        return _emit_error(args, f"Retrieval K must be a positive integer, got {args.k}.")

    try:
        pdf_paths = _resolve_pdf_paths(args.pdf_paths)
    except (FileNotFoundError, ValueError) as err:
        return _emit_error(args, str(err))

    cfg = RAGConfig()
    retriever = Retriever(
        file_paths=pdf_paths,
        persist_directory=args.db_dir or cfg.chroma_dir,
        rebuild_index=bool(args.rebuild),
    )

    analyses = analyze_papers(file_paths=pdf_paths, retriever=retriever, k=args.k)

    if args.json:
        payload = []
        for a in analyses:
            payload.append({
                "document_id": a.document_id,
                "document_name": a.document_name,
                "title": a.title,
                "research_problem": a.research_problem,
                "methodology": a.methodology,
                "major_claims": a.major_claims,
                "evidence": a.evidence,
                "limitations": a.limitations,
                "citations": a.citations,
                "claims": [
                    {
                        "text": c.text,
                        "support_status": c.support_status.value,
                        "citations": c.get_citations(),
                    }
                    for c in a.claims
                ],
            })
        out_str = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(out_str, encoding="utf-8")
            print(f"✓ Analysis JSON saved to {args.output}", file=sys.stderr)
            print(
                json.dumps({
                    "status": "success",
                    "output_file": str(args.output),
                    "paper_count": len(analyses),
                })
            )
        else:
            print(out_str)
        return 0

    lines: List[str] = []
    for idx, a in enumerate(analyses, 1):
        lines.append("=" * 70)
        lines.append(f"Paper Analysis [{idx}/{len(analyses)}]: {a.document_name}")
        lines.append(f"Document ID:      {a.document_id}")
        lines.append(f"Title:            {a.title}")
        lines.append(f"Research Problem: {a.research_problem}")
        lines.append(f"Methodology:      {a.methodology}")
        lines.append(f"Major Claims:     {a.major_claims}")
        lines.append(f"Evidence:         {a.evidence}")
        lines.append(f"Limitations:      {a.limitations}")
        lines.append(f"\nVerified Claims ({len(a.claims)}):")
        for c in a.claims:
            cits = " ".join(c.get_citations())
            lines.append(f"  [{c.support_status.value}] {c.text} {cits}".strip())
        lines.append(f"\nCitations ({len(a.citations)}):")
        for cit in a.citations:
            lines.append(f"  - [{cit.get('document_name')}] Page {cit.get('page')}")
        lines.append("=" * 70 + "\n")

    full_output = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(full_output, encoding="utf-8")
        print(f"✓ Analysis report saved to {args.output}")
    else:
        print(full_output)

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Executes multi-paper comparative synthesis."""
    from retrieval.cross_paper_analysis import analyze_cross_paper
    from retrieval.paper_analysis import analyze_papers
    from retrieval.retriver import Retriever

    if args.k <= 0:
        return _emit_error(args, f"Retrieval K must be a positive integer, got {args.k}.")

    try:
        pdf_paths = _resolve_pdf_paths(args.pdf_paths)
    except (FileNotFoundError, ValueError) as err:
        return _emit_error(args, str(err))

    if len(pdf_paths) < 2 and not args.allow_single:
        return _emit_error(
            args,
            "Compare requires at least 2 papers. Use --allow-single to run on 1 paper.",
        )

    cfg = RAGConfig()
    retriever = Retriever(
        file_paths=pdf_paths,
        persist_directory=args.db_dir or cfg.chroma_dir,
        rebuild_index=bool(args.rebuild),
    )

    analyses = analyze_papers(file_paths=pdf_paths, retriever=retriever, k=args.k)
    cross_result = analyze_cross_paper(analyses)

    if args.json:
        payload = {
            "analyzed_papers": cross_result.analyzed_papers,
            "similarities": cross_result.similarities,
            "differences": cross_result.differences,
            "corroborating_findings": [
                {
                    "finding": f.finding,
                    "supporting_papers": f.supporting_papers,
                    "citations": f.get_citations(),
                }
                for f in cross_result.corroborating_findings_structured
            ],
            "conflicting_findings": [
                {
                    "topic": f.topic,
                    "papers": f.papers,
                    "explanation": f.explanation,
                    "citations": f.get_citations(),
                }
                for f in cross_result.conflicting_findings_structured
            ],
            "research_gaps": [
                {
                    "finding": f.finding,
                    "relevant_papers": f.relevant_papers,
                    "citations": f.get_citations(),
                }
                for f in cross_result.research_gaps_structured
            ],
            "overall_synthesis": cross_result.overall_synthesis,
        }
        out_str = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(out_str, encoding="utf-8")
            print(f"✓ Cross-paper JSON saved to {args.output}", file=sys.stderr)
            print(
                json.dumps({
                    "status": "success",
                    "output_file": str(args.output),
                    "papers_count": len(cross_result.analyzed_papers),
                })
            )
        else:
            print(out_str)
        return 0

    lines: List[str] = [
        "=" * 70,
        "CROSS-PAPER RESEARCH SYNTHESIS",
        "=" * 70,
        f"Analyzed Papers ({len(cross_result.analyzed_papers)}):",
    ]
    for p in cross_result.analyzed_papers:
        lines.append(f" - {p.get('document_name')} ({p.get('document_id')}): {p.get('title')}")
    lines.append(f"\n[1] SIMILARITIES:\n{cross_result.similarities}")
    lines.append(f"\n[2] DIFFERENCES:\n{cross_result.differences}")
    lines.append(f"\n[3] CORROBORATING FINDINGS:\n{cross_result.corroborating_findings}")
    lines.append(f"\n[4] CONFLICTING FINDINGS:\n{cross_result.conflicting_findings}")
    lines.append(f"\n[5] RESEARCH GAPS:\n{cross_result.research_gaps}")
    lines.append(f"\n[6] OVERALL SYNTHESIS:\n{cross_result.overall_synthesis}")
    lines.append("=" * 70)

    full_output = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(full_output, encoding="utf-8")
        print(f"✓ Cross-paper report saved to {args.output}")
    else:
        print(full_output)

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Runs the retrieval benchmark evaluation harness."""
    from evaluation.evaluate import run_evaluation

    if args.k <= 0:
        return _emit_error(args, f"Retrieval K must be a positive integer, got {args.k}.")

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        return _emit_error(args, f"Dataset file not found at {dataset_path}")

    try:
        quiet_mode = bool(args.json)
        metrics = run_evaluation(
            dataset_path=str(dataset_path),
            mode=args.mode,
            target_id_level=args.level,
            max_k=args.k,
            output_json_path=args.output_json,
            quiet=quiet_mode,
        )
        if args.json:
            summary = metrics.to_dict()
            summary["status"] = "success"
            summary["mode"] = args.mode
            summary["target_id_level"] = args.level
            print(json.dumps(summary, indent=2))
        return 0
    except Exception as err:
        return _emit_internal_error(args, f"Error during evaluation: {err}", code=2)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Pre-indexes documents into ChromaDB + BM25 without querying."""
    from retrieval.retriver import Retriever

    try:
        pdf_paths = _resolve_pdf_paths(args.pdf_paths)
    except (FileNotFoundError, ValueError) as err:
        return _emit_error(args, str(err))

    target_stream = sys.stderr if args.json else sys.stdout
    print(f"Indexing {len(pdf_paths)} document(s) into ChromaDB...", file=target_stream)
    for idx, p in enumerate(pdf_paths, 1):
        print(f"  [{idx}] {Path(p).name}", file=target_stream)

    cfg = RAGConfig()
    retriever = Retriever(
        file_paths=pdf_paths,
        persist_directory=args.db_dir or cfg.chroma_dir,
        rebuild_index=bool(args.rebuild),
    )

    indexed = retriever.get_indexed_papers()
    if args.json:
        payload = {
            "status": "success",
            "indexed_papers": indexed,
            "count": len(indexed),
        }
        print(json.dumps(payload, indent=2), file=sys.stdout)
    else:
        print(f"\n✓ Successfully indexed {len(indexed)} paper(s):")
        for item in indexed:
            print(f"  - {item['document_name']} (ID: {item['document_id']})")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launches the PaperTrail / RAGSS FastAPI web server."""
    try:
        import uvicorn
    except ImportError:
        return _emit_error(args, "uvicorn is required to run the web UI. Install with: pip install uvicorn")

    print(f"🚀 Starting PaperTrail / RAGSS Web UI at http://{args.host}:{args.port}")
    uvicorn.run("api.app:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# ----------------------------------------------------------------------
# CLI Parser Definition
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    cfg = RAGConfig()
    parser = argparse.ArgumentParser(
        prog="ragss",
        description="RAGSS: Research Paper Evidence Retrieval & Comparative Synthesis",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Subcommand: query
    p_query = subparsers.add_parser("query", help="Ask a question against indexed documents")
    p_query.add_argument("query", type=str, help="Question to evaluate")
    p_query.add_argument("--docs", nargs="*", default=None, help="Path(s) to PDF papers to index/query")
    p_query.add_argument("--db-dir", type=str, default=None, help="Chroma persistence directory")
    p_query.add_argument("--k", type=int, default=cfg.default_k, help="Number of chunks to retrieve")
    p_query.add_argument(
        "--no-llm",
        action="store_true",
        help="Retrieval-only preview (loads retriever/cross-encoder, skips LLM token generation)",
    )
    p_query.add_argument("--json", action="store_true", help="Output results as JSON")

    # Subcommand: analyze
    p_analyze = subparsers.add_parser("analyze", help="Run structured single or multi-paper analysis")
    p_analyze.add_argument("pdf_paths", nargs="+", help="Path(s) to PDF research papers")
    p_analyze.add_argument("--k", type=int, default=cfg.default_k, help="Chunks to retrieve per paper")
    p_analyze.add_argument("--db-dir", type=str, default=None, help="Chroma persistence directory")
    p_analyze.add_argument("--rebuild", action="store_true", help="Force rebuild Chroma vector index")
    p_analyze.add_argument("--output", type=str, default=None, help="Path to save report output")
    p_analyze.add_argument("--json", action="store_true", help="Output results as JSON")

    # Subcommand: compare
    p_compare = subparsers.add_parser("compare", help="Run cross-paper comparative synthesis")
    p_compare.add_argument("pdf_paths", nargs="+", help="Paths to 2 or more PDF research papers")
    p_compare.add_argument("--k", type=int, default=cfg.default_k, help="Chunks to retrieve per paper")
    p_compare.add_argument("--db-dir", type=str, default=None, help="Chroma persistence directory")
    p_compare.add_argument("--rebuild", action="store_true", help="Force rebuild Chroma vector index")
    p_compare.add_argument("--allow-single", action="store_true", help="Allow running on a single paper")
    p_compare.add_argument("--output", type=str, default=None, help="Path to save report output")
    p_compare.add_argument("--json", action="store_true", help="Output results as JSON")

    # Subcommand: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run the retrieval benchmark harness")
    p_eval.add_argument("--mode", choices=["fixture", "live"], default="fixture", help="Evaluation mode")
    p_eval.add_argument(
        "--dataset",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "retrieval_dataset.jsonl"),
        help="Path to evaluation dataset JSONL",
    )
    p_eval.add_argument("--level", choices=["parent", "chunk"], default="parent", help="Target ID level")
    p_eval.add_argument("--k", type=int, default=5, help="Maximum retrieval cutoff K")
    p_eval.add_argument("--output-json", type=str, default=None, help="Path to write JSON results")
    p_eval.add_argument("--json", action="store_true", help="Output benchmark summary as JSON to stdout")

    # Subcommand: ingest
    p_ingest = subparsers.add_parser("ingest", help="Index PDF documents without querying")
    p_ingest.add_argument("pdf_paths", nargs="+", help="Path(s) to PDF research papers")
    p_ingest.add_argument("--db-dir", type=str, default=None, help="Chroma persistence directory")
    p_ingest.add_argument("--rebuild", action="store_true", help="Force rebuild Chroma vector index")
    p_ingest.add_argument("--json", action="store_true", help="Output indexing status as JSON")

    # Subcommand: serve
    p_serve = subparsers.add_parser("serve", help="Start the PaperTrail web UI and API server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    p_serve.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    p_serve.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # Standard argparse exit: 0 for --help, 2 for CLI syntax errors (unrecognized arguments)
        code = exc.code if isinstance(exc.code, int) else 2
        return code

    if not args.command:
        parser.print_help()
        return 0

    command_map = {
        "query": cmd_query,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "evaluate": cmd_evaluate,
        "ingest": cmd_ingest,
        "serve": cmd_serve,
    }

    handler = command_map.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except ValueError as err:
        return _emit_error(args, f"Configuration / validation error: {err}", code=1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        if getattr(args, "json", False):
            print(json.dumps({"status": "cancelled", "error": "Operation cancelled by user.", "exit_code": 130}))
        return 130
    except Exception as err:
        logger.exception(f"Unhandled error in subcommand '{args.command}': {err}")
        return _emit_internal_error(args, str(err), code=2)


if __name__ == "__main__":
    sys.exit(main())
