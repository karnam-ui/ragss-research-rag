# RAGSS: Research Paper Evidence Retrieval & Comparative Synthesis

RAGSS is a multi-stage Retrieval-Augmented Generation (RAG) system tailored for scientific literature and technical documents. It couples hierarchical parent-child document chunking with hybrid dense-sparse retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and structured evidence provenance tracking.

---

## 1. Problem Statement & Scope

Modern Large Language Models (LLMs) struggle with scientific document analysis due to:
1. **Context Fragmentation:** Standard fixed-size chunking splits complex paragraphs, losing critical surrounding mathematical or methodology context.
2. **Vocabulary Mismatch vs. Semantic Drift:** Dense vector embeddings alone often miss exact terminology, acronyms, or symbols, while pure keyword search (BM25) fails on paraphrased queries.
3. **Ungrounded Citations:** LLMs routinely fabricate inline citations or attribute claims to irrelevant sections.
4. **Lack of Measurement:** Most RAG prototypes lack evaluation harnesses to objectively measure whether the retriever finds the right evidence or prevents hallucination on unanswerable questions.

RAGSS addresses these challenges through a transparent, measurable pipeline with deterministic offline evaluation.

---

## 2. Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion ["1. Hierarchical Ingestion & Indexing"]
        PDF["PDF Research Papers"] -->|PyMuPDFLoader| Parents["Parent Documents (Full Pages)"]
        Parents -->|RecursiveCharacterTextSplitter<br>chunk_size=200, overlap=20| Children["Child Chunks<br>(doc_id, parent_id, chunk_id)"]
        Children -->|BAAI/bge-small-en-v1.5| Chroma["ChromaDB<br>(Dense Index)"]
        Children -->|Regex Tokenizer| BM25["BM25Okapi<br>(Sparse Index)"]
        Parents -->|In-Memory Map| ParentStore["Parent Document Store"]
    end

    subgraph Retrieval ["2. Hybrid Retrieval & Reranking"]
        Query["User Query"] -->|similarity_search k=8| Chroma
        Query -->|bm25.get_top_n k=8| BM25
        Chroma -->|Top 8 Dense Candidates| RRF["Reciprocal Rank Fusion<br>score = 1 / (60 + rank)"]
        BM25 -->|Top 8 Sparse Candidates| RRF
        RRF -->|Fused Candidates| CrossEncoder["Cross-Encoder Reranking<br>(ms-marco-MiniLM-L-6-v2)"]
        CrossEncoder -->|Top Child Chunks| Expansion["Parent Document Expansion<br>parent_by_id.get(parent_id)"]
        Expansion --> Evidence["Expanded Parent Evidence (Top K)"]
    end

    subgraph Generation ["3. Grounding & Synthesis"]
        Evidence -->|PAPER_ANALYSIS_PROMPT| Analyzer["PaperAnalyzer<br>(Qwen2.5-1.5B-Instruct)"]
        Analyzer --> SingleAnalysis["PaperAnalysis Object<br>(Title, Problem, Claims, Limitations, Citations)"]
        SingleAnalysis -->|CROSS_PAPER_PROMPT| CrossAnalyzer["CrossPaperAnalyzer"]
        CrossAnalyzer --> CrossAnalysis["CrossPaperAnalysis Object<br>(Similarities, Differences, Gaps, Conflicts)"]
    end
```

### Key Architectural Decisions
- **Hierarchical Parent-Child Representation:** Retrieval indexes granular 200-character child chunks to preserve high vector density and semantic focus. Upon retrieval and reranking, child chunks expand to full parent pages to provide the LLM with complete surrounding context.
- **Hybrid Retrieval (Dense + Sparse):** Dense embeddings (`BAAI/bge-small-en-v1.5`, 384 dimensions) capture conceptual similarity; `BM25Okapi` captures exact terminology and model identifiers.
- **Reciprocal Rank Fusion (RRF):** Fuses dense and sparse rank lists using standard RRF ($k=60$ constant) without requiring score normalization.
- **Cross-Encoder Reranking:** Applies `cross-encoder/ms-marco-MiniLM-L-6-v2` over fused candidates to perform full cross-attention over query-passage pairs.
- **Strict Evidence Provenance:** All returned passages and generated claims are backed by strongly-typed `Evidence` objects preserving `document_id`, `document_name`, `page` (1-indexed for display), `chunk_id`, and `parent_id`.

---

## 3. Implemented Capabilities

| Component | Status | Details |
| :--- | :--- | :--- |
| **Hierarchical PDF Ingestion** | Implemented | PyMuPDF page-level extraction + recursive character splitting into child chunks. |
| **Hybrid Dense-Sparse Index** | Implemented | ChromaDB vector store + BM25Okapi lexical corpus with per-document filtering. |
| **Reciprocal Rank Fusion (RRF)** | Implemented | Fuses dual rankings into a unified candidate pool. |
| **Cross-Encoder Reranking** | Implemented | Pre-trained MS-MARCO cross-encoder reranking fused children before expansion. |
| **Parent Document Expansion** | Implemented | Resolves top reranked child chunks to full parent pages, with fallback to child if parent missing. |
| **Structured Provenance Data Model** | Implemented | `Evidence`, `Claim`, `CorroboratingFinding`, `ConflictingFinding`, `ResearchGap`, `EvidenceAggregator`. |
| **Single-Paper Extraction** | Implemented | Extracts structured sections (Title, Research Problem, Methodology, Claims, Evidence, Limitations). |
| **Cross-Paper Synthesis** | Implemented | Compares multiple papers for corroborations, conflicts, and gaps. |
| **Offline Evaluation Harness** | Implemented | Evaluates Recall@K, MRR, Hit@K, and unanswerable query abstention. |
| **Fast Deterministic Test Suite** | Implemented | 110 automated unit/regression tests running in ~2.8 seconds offline. |
| **Unified Production CLI (`main.py`)** | Implemented | Subcommands: `query`, `analyze`, `compare`, `evaluate`, and `ingest`. |

---

## 4. Installation & Setup

### Prerequisites
- Python 3.10–3.13
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/your-username/RAGSS.git
cd RAGSS

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 5. Running the Automated Test Suite

The test suite separates **fast offline deterministic tests** from **heavy live-model integration tests** via `pytest.ini`.

### Fast Offline Unit Tests (Default)
Executes all 110 offline unit tests (CLI error contracts, configuration dynamic precedence and validation, metrics correctness, hybrid retrieval logic, parent-child expansion, and evidence provenance) in **under 3 seconds without downloading model weights**:
```bash
python -m pytest
```

### Live Model Integration Tests (Optional)
Executes end-to-end integration tests using local Hugging Face model weights (`bge-small-en-v1.5`, `Qwen2.5-1.5B-Instruct`):
```bash
python -m pytest -m integration
```

---

## 6. Retrieval Evaluation Harness

The evaluation harness evaluates retrieval quality against a versioned benchmark dataset ([`evaluation/retrieval_dataset.jsonl`](evaluation/retrieval_dataset.jsonl)) with 12 annotated queries:
- **Single-Paper Questions (3):** Focused on core architectural contributions.
- **Specific Passage Questions (3):** Exact mathematical formulas and training objectives.
- **Multi-Paper Questions (2):** Requires retrieving evidence across multiple distinct papers.
- **Confusable Terminology (1):** Distinguishing parametric from non-parametric retrieval memory.
- **Unanswerable / Out-of-Corpus Questions (3):** Tests system behavior on unindexed domains (quantum error correction, banking rates, protein biology).

### A. Fast Offline Fixture Mode (Recommended for CI/CD)
Runs the complete benchmark through the deterministic fixture retriever in `< 0.1` seconds:
```bash
python evaluation/evaluate.py --mode fixture
```

### B. Live Corpus Evaluation Mode
Indexes the 3 sample research papers and runs live hybrid retrieval with cross-encoder reranking:
```bash
python evaluation/evaluate.py --mode live --level parent --k 5
```

Export detailed evaluation JSON:
```bash
python evaluation/evaluate.py --mode fixture --output-json evaluation/benchmark_results.json
```

---

## 7. Actual Evaluation Benchmark Results

The following verified metrics were produced by running the evaluation harness against the annotated benchmark dataset:

### Live Evaluation Mode (`--mode live --level parent --k 5`)
*Evaluated on real indexed research papers (`paper1_transformers.pdf`, `paper2_bert.pdf`, `paper3_rag.pdf`) with live `bge-small-en-v1.5` embeddings, BM25, and MS-MARCO Cross-Encoder:*

| Metric | Measured Value | Description |
| :--- | :--- | :--- |
| **Total Evaluated Queries** | **12** | Complete benchmark query set. |
| **Answerable Queries** | **9** | Queries with ground-truth evidence in corpus. |
| **Unanswerable Queries** | **3** | Out-of-corpus queries (strictly separated from recall). |
| **Mean Recall@1** | **0.8889** | 8 of 9 queries retrieved all expected targets at rank 1. |
| **Mean Recall@3** | **1.0000** | 100% of ground-truth target documents retrieved in top 3. |
| **Mean Recall@5** | **1.0000** | 100% of ground-truth target documents retrieved in top 5. |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | Every answerable query found its first relevant document at rank 1. |
| **Zero-Hit Queries (Miss Rate)** | **0 / 9 (0%)** | Zero retrieval misses on answerable queries. |
| **Unanswerable Queries Handled** | **3 / 3 (100%)** | All out-of-corpus queries tracked without polluting recall scores. |
| **Average Query Latency** | **~35 ms** | Per-query retrieval and cross-encoder inference latency. |

### Deterministic Fixture Mode (`--mode fixture --level chunk`)
*Evaluated via controlled offline keyword/chunk mapping for regression testing:*
- **Mean Recall@1:** 0.8889
- **Mean Recall@3:** 1.0000
- **Mean Recall@5:** 1.0000
- **Mean MRR:** 1.0000
- **Execution Time:** < 0.05 seconds

---

## 8. Unified Command-Line Interface (`main.py`)

RAGSS exposes a unified, production-ready CLI entrypoint via `main.py` (command `ragss`) with 5 dedicated subcommands.

### A. Grounded Querying (`ragss query`)
Ask natural-language questions against indexed documents with full evidence provenance:
```bash
# Query indexed documents with local LLM generation
python main.py query "How does scaled dot-product attention compute weights?" --k 4

# Fast retrieval preview mode: runs embedding inference, BM25, RRF, cross-encoder reranking,
# and parent expansion while bypassing the generative LLM (Qwen2.5-1.5B) and token decoding
python main.py query "What is MLM in BERT?" --no-llm

# Structured JSON output for downstream tooling / automation
python main.py query "Explain non-parametric memory in RAG" --json
```

### B. Structured Paper Analysis (`ragss analyze`)
Extract structured sections (Problem, Methodology, Major Claims, Empirical Evidence, Limitations) and verified `Claim` objects:
```bash
# Analyze a single paper or multiple papers
python main.py analyze tests/sample_papers/paper1_transformers.pdf --k 4

# Export analysis results as structured JSON
python main.py analyze tests/sample_papers/paper1_transformers.pdf tests/sample_papers/paper2_bert.pdf --output analysis.json
```

### C. Cross-Paper Comparative Synthesis (`ragss compare`)
Perform multi-document comparative analysis identifying shared findings, conflicting methodologies, and research gaps:
```bash
# Compare two or more research papers
python main.py compare tests/sample_papers/paper1_transformers.pdf tests/sample_papers/paper2_bert.pdf tests/sample_papers/paper3_rag.pdf --output comparison.json
```

### D. Benchmark Evaluation Harness (`ragss evaluate`)
Run the 12-query retrieval benchmark suite:
```bash
# Fast deterministic offline fixture mode (< 0.1s)
python main.py evaluate --mode fixture --k 5

# Live corpus retrieval evaluation
python main.py evaluate --mode live --k 5 --output-json evaluation_results.json

# Machine-readable JSON output to stdout
python main.py evaluate --mode fixture --json
```

### E. Standalone Ingestion (`ragss ingest`)
Index documents into ChromaDB and initialize BM25 lexical structures without immediately executing queries:
```bash
python main.py ingest tests/sample_papers/ --json
```

### F. Legacy Compatibility Entry Points
For backward compatibility with existing scripts, legacy entry points remain supported:
```bash
# Legacy multi-paper script
python -m retrieval.paper_analysis tests/sample_papers/paper1_transformers.pdf tests/sample_papers/paper2_bert.pdf --k 4

# Banking multi-year rate analysis demo
python run_bank_query.py --docs-dir tests/bank_docs --query "what is the roi of savings acc of 22-23" --k 3
```

---

## 9. Configuration & Environment Variables

All pipeline parameters are centralized in [`retrieval/config.py`](retrieval/config.py) via `RAGConfig`. The configuration follows dynamic precedence evaluated at instantiation time:
`Explicit constructor args > Environment variables (RAGSS_*) > Documented defaults`

| Environment Variable | Default | Purpose |
| :--- | :--- | :--- |
| `RAGSS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Dense embedding model checkpoint |
| `RAGSS_CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `RAGSS_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Local generation LLM |
| `RAGSS_CHUNK_SIZE` | `200` | Child chunk character size (> 0) |
| `RAGSS_CHUNK_OVERLAP` | `20` | Child chunk character overlap (>= 0, < size) |
| `RAGSS_CHROMA_DIR` | `./chroma_db` | Persistent ChromaDB storage path |
| `RAGSS_COLLECTION_NAME` | `research_papers` | ChromaDB collection name |
| `RAGSS_RETRIEVAL_K` | `4` | Default retrieved passages count (> 0) |
| `RAGSS_RRF_K` | `60` | Reciprocal Rank Fusion constant (> 0) |
| `RAGSS_SUPPORT_THRESHOLD` | `0.5` | Minimum token overlap for `SUPPORTED` claim |
| `RAGSS_PARTIAL_THRESHOLD` | `0.25` | Minimum token overlap for `PARTIALLY_SUPPORTED` |
| `RAGSS_CROSS_FINDING_THRESHOLD`| `0.15`| Minimum overlap for cross-paper finding evidence |

Example runtime override:
```bash
RAGSS_RETRIEVAL_K=6 RAGSS_SUPPORT_THRESHOLD=0.6 python main.py query "What is scaled attention?" --no-llm
```

---

## 10. Exit Code Standards & Stream Isolation

### Exit Code Contract
The CLI strictly adheres to standard POSIX process exit conventions:
- **`0`**: Successful execution of the requested operation or `--help`.
- **`1`**: Application-level domain/validation error (e.g., non-existent file, empty query text, non-PDF file, comparing $< 2$ papers without `--allow-single`, non-positive $K$, or invalid configuration values). Informative diagnostic written to `stderr`.
- **`2`**: Command-line syntax error (unrecognized argument flags or missing option parameters raised by `argparse`) OR unhandled internal system/database exceptions.
- **`130`**: Interrupted by user via SIGINT / `KeyboardInterrupt` (`128 + 2`).

### Machine-Readable Stream Isolation (`--json`)
When `--json` is active:
- **`stderr`**: Strictly receives all human-readable diagnostic messages, warnings, indexing progress lines, and error notices.
- **`stdout`**: Strictly reserved for valid, parseable JSON payloads in both success and failure cases. If an error occurs, `stdout` outputs `{"status": "error", "error": "...", "exit_code": 1}` so downstream automated tools never encounter `JSONDecodeError`.

---

## 11. Known Limitations & Engineering Boundaries

1. **In-Memory Parent Document Mapping:** Parent documents are currently held in memory in `self.parent_by_id`. Re-opening a persisted ChromaDB database across process boundaries requires either re-ingesting the PDFs or persisting parent text in metadata.
2. **Page-Based Chunking Boundaries:** Parents are segmented by raw PDF page boundaries rather than semantic markdown/section headers. If a paragraph spans page boundaries, context may be split.
3. **No NLI Entailment Scoring:** Citations are deterministically mapped to retrieved evidence passages, but token-level semantic entailment between each generated claim and cited evidence is not yet scored with an NLI classifier.
4. **LangGraph Workflow Integration:** `workflows/workflow.py` contains an architectural prototype with mock nodes; active production retrieval runs directly through `Retriever` and LCEL pipelines in `paper_analysis.py`.
