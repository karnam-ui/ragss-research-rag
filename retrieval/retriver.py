import sys
import re
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.embeddingss import Embed

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self,
        file_paths=None,
        file_path=None,
        persist_directory=None,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name="research_papers",
        rebuild_index=False,
        embedder=None,
        vector_store=None,
        cross_encoder=None,
        parents=None,
        documents=None,
    ):
        target_paths = file_paths if file_paths is not None else file_path

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = Embed(target_paths)

        if parents is not None:
            self.parents = parents
        else:
            self.parents, _ = self.embedder.Embedchunks()

        self.parent_by_id = {
            document.metadata["parent_id"]: document
            for document in self.parents
            if document.metadata.get("parent_id")
        }

        self.persist_directory = Path(persist_directory or PROJECT_ROOT / "chroma_db")
        self.collection_name = collection_name

        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(self.persist_directory),
                embedding_function=getattr(self.embedder, "embedding_model", None),
            )

            if rebuild_index or self._needs_indexing():
                self.index_documents()

        if documents is not None:
            self.documents = documents
        else:
            self.documents = self._load_documents()

        if self.documents:
            tokenized_corpus = [self._tokenize(document.page_content) for document in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._doc_token_sets = [set(tokens) for tokens in tokenized_corpus]
        else:
            self.bm25 = None
            self._doc_token_sets = []

        if cross_encoder is not None:
            self.cross_encoder = None if cross_encoder is False else cross_encoder
        elif cross_encoder_model:
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        else:
            self.cross_encoder = None

    def index_documents(self):
        if not hasattr(self.embedder, "Embedchunks"):
            return
        _, children = self.embedder.Embedchunks()
        existing_ids = self.vector_store.get().get("ids", [])
        if existing_ids:
            self.vector_store.delete(ids=existing_ids)
        if children:
            self.vector_store = Chroma.from_documents(
                documents=children,
                embedding=self.embedder.embedding_model,
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
            )

    def rebuild_index(self):
        """Safely clears and re-indexes all documents into the shared Chroma collection."""
        self.index_documents()
        self.documents = self._load_documents()
        if self.documents:
            tokenized_corpus = [self._tokenize(document.page_content) for document in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._doc_token_sets = [set(tokens) for tokens in tokenized_corpus]
        else:
            self.bm25 = None
            self._doc_token_sets = []

    def _needs_indexing(self):
        stored = self.vector_store.get(include=["metadatas"])
        metadatas = stored.get("metadatas", [])
        return not metadatas or any(
            metadata is None
            or "parent_id" not in metadata
            or "document_id" not in metadata
            or "chunk_id" not in metadata
            for metadata in metadatas
        )

    def _load_documents(self):
        stored = self.vector_store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(stored["documents"], stored["metadatas"])
        ]

    @staticmethod
    def _tokenize(text):
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def get_indexed_papers(self):
        """Returns a list of unique paper metadata dictionaries [{'document_id': ..., 'document_name': ...}]."""
        seen_ids = set()
        papers = []
        for doc in self.documents:
            doc_id = doc.metadata.get("document_id")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                papers.append({
                    "document_id": doc_id,
                    "document_name": doc.metadata.get("document_name", "Unknown"),
                })
        return papers

    def retrieve(
        self,
        query: str,
        k: int = 4,
        candidate_k: Optional[int] = None,
        document_id: Optional[str] = None,
        rrf_k: int = 60,
    ) -> List[Document]:
        """
        Executes hybrid retrieval (dense vector similarity + lexical BM25), reciprocal rank fusion (RRF),
        optional cross-encoder reranking, and hierarchical parent-child expansion.

        Guarantees:
        - If query is empty, whitespace-only, or non-string: returns [] immediately.
        - If k <= 0: returns [] immediately.
        - candidate_k is clamped to max(candidate_k, k) so candidate pool is never smaller than requested k.
        - If document_id is provided, both dense and lexical branches filter strictly, and parent expansion
          preserves the filter invariant.
        - BM25 candidates strictly require lexical term overlap with query tokens (no zero-overlap hallucinations).
        - If cross-encoder fails during reranking, gracefully logs a warning and falls back to RRF rank order.
        - Orphan chunks without resolvable parents fall back safely without blocking subsequent orphans.
        """
        docs, _ = self.retrieve_with_diagnostics(
            query=query,
            k=k,
            candidate_k=candidate_k,
            document_id=document_id,
            rrf_k=rrf_k,
        )
        return docs

    def retrieve_with_diagnostics(
        self,
        query: str,
        k: int = 4,
        candidate_k: Optional[int] = None,
        document_id: Optional[str] = None,
        rrf_k: int = 60,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Executes hybrid retrieval and captures real stage-by-stage diagnostics:
        - Timing per stage (dense, BM25, RRF, rerank, parent expansion, total).
        - Candidate counts and flags per stage.
        - Detailed chunk provenance: dense score & rank, BM25 score & rank,
          RRF score & rank, cross-encoder score & rank, final rank, parent expansion state.
        """
        start_time = time.perf_counter()

        diagnostics: Dict[str, Any] = {
            "query": query,
            "stages": {
                "dense": {"ran": False, "retrieved_count": 0, "timing_ms": 0.0, "error": None},
                "bm25": {"ran": False, "retrieved_count": 0, "timing_ms": 0.0, "error": None},
                "rrf": {"ran": False, "fused_count": 0, "timing_ms": 0.0, "k": rrf_k},
                "cross_encoder": {"ran": False, "reranked_count": 0, "fallback": False, "timing_ms": 0.0, "error": None},
                "parent_expansion": {"ran": False, "expanded_count": 0, "fallback_count": 0, "timing_ms": 0.0},
            },
            "filters": {
                "document_id": document_id,
                "k": k,
                "candidate_k": candidate_k,
                "rrf_k": rrf_k,
            },
            "timings": {
                "dense_ms": 0.0,
                "bm25_ms": 0.0,
                "rrf_ms": 0.0,
                "rerank_ms": 0.0,
                "parent_expansion_ms": 0.0,
                "total_ms": 0.0,
            },
            "abstention_reason": None,
            "chunks": [],
        }

        # 1. Input validation & contract boundaries
        if query is None or not isinstance(query, str) or not query.strip():
            diagnostics["abstention_reason"] = "Empty, whitespace-only, or invalid query."
            diagnostics["timings"]["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return [], diagnostics

        if k is None:
            k = 4
        try:
            k = int(k)
        except (ValueError, TypeError):
            diagnostics["abstention_reason"] = "Invalid k value."
            diagnostics["timings"]["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return [], diagnostics
        if k <= 0:
            diagnostics["abstention_reason"] = f"k must be greater than 0 (got {k})."
            diagnostics["timings"]["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return [], diagnostics

        if candidate_k is None or candidate_k <= 0:
            candidate_k = max(8, k * 2)
        else:
            try:
                candidate_k = int(candidate_k)
            except (ValueError, TypeError):
                candidate_k = max(8, k * 2)
        candidate_k = max(candidate_k, k)

        if document_id is not None:
            document_id = str(document_id).strip() or None

        diagnostics["filters"]["document_id"] = document_id
        diagnostics["filters"]["k"] = k
        diagnostics["filters"]["candidate_k"] = candidate_k

        # 2. Dense semantic retrieval
        t_dense_start = time.perf_counter()
        search_kwargs = {}
        if document_id:
            search_kwargs["filter"] = {"document_id": document_id}

        dense_scores: Dict[Any, float] = {}
        dense_error = None
        try:
            if hasattr(self.vector_store, "similarity_search_with_score"):
                try:
                    scored_docs = self.vector_store.similarity_search_with_score(query, k=candidate_k, **search_kwargs)
                    semantic_documents = [doc for doc, _ in scored_docs]
                    dense_scores = {self._document_key(doc): float(sc) for doc, sc in scored_docs}
                except Exception:
                    semantic_documents = self.vector_store.similarity_search(query, k=candidate_k, **search_kwargs)
            else:
                semantic_documents = self.vector_store.similarity_search(query, k=candidate_k, **search_kwargs)
        except Exception as e:
            logger.warning("Vector store similarity search failed (%s). Continuing with empty dense pool.", e)
            dense_error = str(e)
            semantic_documents = []

        if document_id:
            semantic_documents = [
                doc for doc in semantic_documents
                if doc.metadata.get("document_id") == document_id
            ]

        dense_ranks: Dict[Any, int] = {
            self._document_key(doc): rank
            for rank, doc in enumerate(semantic_documents, start=1)
        }
        t_dense_ms = (time.perf_counter() - t_dense_start) * 1000.0
        diagnostics["stages"]["dense"] = {
            "ran": True,
            "retrieved_count": len(semantic_documents),
            "timing_ms": round(t_dense_ms, 2),
            "error": dense_error,
        }
        diagnostics["timings"]["dense_ms"] = round(t_dense_ms, 2)

        # 3. Lexical BM25 retrieval (strictly requiring lexical term overlap)
        t_bm25_start = time.perf_counter()
        bm25_indexes = []
        bm25_scores_map: Dict[Any, float] = {}
        bm25_ranks: Dict[Any, int] = {}
        bm25_error = None

        if self.bm25 is not None and self.documents:
            try:
                tokenized_query = self._tokenize(query)
                query_token_set = set(tokenized_query)
                if query_token_set:
                    scores = self.bm25.get_scores(tokenized_query)
                    if document_id:
                        candidate_indexes = [
                            idx for idx, doc in enumerate(self.documents)
                            if doc.metadata.get("document_id") == document_id
                            and bool(query_token_set.intersection(self._doc_token_sets[idx]))
                        ]
                    else:
                        candidate_indexes = [
                            idx for idx in range(len(self.documents))
                            if bool(query_token_set.intersection(self._doc_token_sets[idx]))
                        ]

                    if candidate_indexes:
                        sorted_indexes = sorted(
                            candidate_indexes, key=lambda idx: scores[idx], reverse=True
                        )
                        bm25_indexes = sorted_indexes[:candidate_k]
                        for rank, idx in enumerate(bm25_indexes, start=1):
                            doc_key = self._document_key(self.documents[idx])
                            bm25_scores_map[doc_key] = float(scores[idx])
                            bm25_ranks[doc_key] = rank
            except Exception as e:
                logger.warning("BM25 retrieval failed (%s).", e)
                bm25_error = str(e)

        lexical_documents = [self.documents[index] for index in bm25_indexes]
        t_bm25_ms = (time.perf_counter() - t_bm25_start) * 1000.0
        diagnostics["stages"]["bm25"] = {
            "ran": bool(self.bm25 is not None and self.documents),
            "retrieved_count": len(lexical_documents),
            "timing_ms": round(t_bm25_ms, 2),
            "error": bm25_error,
        }
        diagnostics["timings"]["bm25_ms"] = round(t_bm25_ms, 2)

        # 4. Reciprocal Rank Fusion (RRF)
        t_rrf_start = time.perf_counter()
        rrf_scores: Dict[Any, float] = {}
        document_by_key: Dict[Any, Document] = {}
        for ranked_documents in (semantic_documents, lexical_documents):
            for rank, document in enumerate(ranked_documents, start=1):
                key = self._document_key(document)
                if key not in document_by_key:
                    document_by_key[key] = document
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (float(rrf_k) + float(rank))

        # Stable sort descending by RRF score (Timsort preserves order of appearance for ties)
        fused_documents = [
            document_by_key[key]
            for key, _ in sorted(
                rrf_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        rrf_ranks: Dict[Any, int] = {
            self._document_key(doc): rank
            for rank, doc in enumerate(fused_documents, start=1)
        }
        t_rrf_ms = (time.perf_counter() - t_rrf_start) * 1000.0
        diagnostics["stages"]["rrf"] = {
            "ran": True,
            "fused_count": len(fused_documents),
            "timing_ms": round(t_rrf_ms, 2),
            "k": rrf_k,
        }
        diagnostics["timings"]["rrf_ms"] = round(t_rrf_ms, 2)

        if not fused_documents:
            diagnostics["abstention_reason"] = "No matching chunks found by dense or lexical search."
            diagnostics["timings"]["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)
            return [], diagnostics

        # 5. Optional Cross-Encoder reranking with fault tolerance
        t_ce_start = time.perf_counter()
        ce_scores: Dict[Any, float] = {}
        ce_ranks: Dict[Any, int] = {}
        ce_fallback = False
        ce_error = None

        if self.cross_encoder is not None:
            try:
                rerank_scores = self.cross_encoder.predict(
                    [(query, document.page_content) for document in fused_documents]
                )
                reranked_children = sorted(
                    zip(rerank_scores, fused_documents),
                    key=lambda item: item[0],
                    reverse=True,
                )
                for rank, (score_val, doc) in enumerate(reranked_children, start=1):
                    doc_key = self._document_key(doc)
                    ce_scores[doc_key] = float(score_val)
                    ce_ranks[doc_key] = rank
            except Exception as e:
                logger.warning("Cross-encoder reranking failed (%s). Falling back to RRF rank order.", e)
                ce_error = str(e)
                ce_fallback = True
                reranked_children = [(1.0, doc) for doc in fused_documents]
        else:
            reranked_children = [(1.0, doc) for doc in fused_documents]

        t_ce_ms = (time.perf_counter() - t_ce_start) * 1000.0
        diagnostics["stages"]["cross_encoder"] = {
            "ran": (self.cross_encoder is not None),
            "reranked_count": len(reranked_children) if self.cross_encoder is not None else 0,
            "fallback": ce_fallback,
            "timing_ms": round(t_ce_ms, 2),
            "error": ce_error,
        }
        diagnostics["timings"]["rerank_ms"] = round(t_ce_ms, 2)

        # 6. Hierarchical parent expansion with strict deduplication & filter validation
        t_pe_start = time.perf_counter()
        parent_documents: List[Document] = []
        seen_parent_ids: Set[str] = set()
        seen_chunk_keys: Set[Any] = set()

        chunk_diagnostics: List[Dict[str, Any]] = []
        final_rank = 0
        expanded_count = 0
        fallback_count = 0

        for _, child in reranked_children:
            parent_id = child.metadata.get("parent_id")
            parent = self.parent_by_id.get(parent_id) if parent_id else None
            child_key = self._document_key(child)

            is_selected = False
            was_expanded = False

            if len(parent_documents) < k:
                if parent is not None:
                    # If document_id filter is active, enforce on parent document
                    if not (document_id and parent.metadata.get("document_id") != document_id):
                        if parent_id not in seen_parent_ids:
                            seen_parent_ids.add(parent_id)
                            parent_documents.append(parent)
                            is_selected = True
                            was_expanded = True
                            expanded_count += 1
                else:
                    # Fallback: orphan chunk without resolvable parent
                    if not (document_id and child.metadata.get("document_id") != document_id):
                        if child_key not in seen_chunk_keys:
                            seen_chunk_keys.add(child_key)
                            parent_documents.append(child)
                            is_selected = True
                            was_expanded = False
                            fallback_count += 1

            if is_selected:
                final_rank += 1
                current_final_rank = final_rank
            else:
                current_final_rank = None

            chunk_diagnostics.append({
                "chunk_id": child.metadata.get("chunk_id", str(child_key)),
                "parent_id": parent_id,
                "document_id": child.metadata.get("document_id"),
                "document_name": child.metadata.get("document_name"),
                "source": child.metadata.get("source"),
                "page": child.metadata.get("page"),
                "chunk_text": child.page_content,
                "dense_rank": dense_ranks.get(child_key),
                "dense_score": dense_scores.get(child_key),
                "bm25_rank": bm25_ranks.get(child_key),
                "bm25_score": bm25_scores_map.get(child_key),
                "rrf_rank": rrf_ranks.get(child_key),
                "rrf_score": rrf_scores.get(child_key),
                "cross_encoder_rank": ce_ranks.get(child_key),
                "cross_encoder_score": ce_scores.get(child_key),
                "final_rank": current_final_rank,
                "expanded": was_expanded,
                "parent_content": parent.page_content if (was_expanded and parent) else None,
                "metadata": dict(child.metadata),
            })

        t_pe_ms = (time.perf_counter() - t_pe_start) * 1000.0
        diagnostics["stages"]["parent_expansion"] = {
            "ran": True,
            "expanded_count": expanded_count,
            "fallback_count": fallback_count,
            "timing_ms": round(t_pe_ms, 2),
        }
        diagnostics["timings"]["parent_expansion_ms"] = round(t_pe_ms, 2)
        diagnostics["chunks"] = chunk_diagnostics
        diagnostics["timings"]["total_ms"] = round((time.perf_counter() - start_time) * 1000.0, 2)

        return parent_documents, diagnostics

    @staticmethod
    def _document_key(document):
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id:
            return chunk_id
        return (
            document.metadata.get("document_id"),
            document.metadata.get("source"),
            document.metadata.get("page"),
            document.page_content,
        )
