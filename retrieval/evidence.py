import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


@dataclass
class Evidence:
    evidence_text: str
    document_id: str
    document_name: str
    page: int
    chunk_id: Optional[str] = None
    parent_id: Optional[str] = None
    support_status: SupportStatus = SupportStatus.NOT_ENOUGH_EVIDENCE

    @classmethod
    def from_document(
        cls,
        doc: Document,
        support_status: SupportStatus = SupportStatus.NOT_ENOUGH_EVIDENCE,
    ) -> "Evidence":
        if not isinstance(doc, Document):
            raise TypeError(f"Expected langchain_core.documents.Document, got {type(doc)}")

        raw_page = doc.metadata.get("page", 0)
        # Consistent with project convention: raw PDF pages are 0-indexed, display citations are 1-indexed
        try:
            page_num = int(raw_page) + 1
        except (ValueError, TypeError):
            page_num = 1

        return cls(
            evidence_text=doc.page_content,
            document_id=doc.metadata.get("document_id", "unknown_doc"),
            document_name=doc.metadata.get("document_name", "unknown_document.pdf"),
            page=page_num,
            chunk_id=doc.metadata.get("chunk_id"),
            parent_id=doc.metadata.get("parent_id"),
            support_status=support_status,
        )


def build_citation(evidence: Evidence, style: str = "bracket") -> str:
    """
    Deterministically formats a citation string from an Evidence object.
    Strictly uses Evidence.document_name and Evidence.page.
    Rejects raw strings or ungrounded LLM inputs.
    """
    if not isinstance(evidence, Evidence):
        raise TypeError(
            f"build_citation strictly requires an Evidence instance, got {type(evidence)}. "
            "Arbitrary LLM citation strings are not permitted."
        )

    if style == "inline":
        return f"{evidence.document_name} · p.{evidence.page}"
    return f"[{evidence.document_name}, p.{evidence.page}]"


def build_citations(evidence_list: List[Evidence], style: str = "bracket") -> List[str]:
    """
    Deterministically formats and deduplicates citation strings across a list of Evidence objects,
    preserving order of appearance.
    """
    seen: Set[str] = set()
    citations: List[str] = []
    for ev in evidence_list:
        cit = build_citation(ev, style=style)
        if cit not in seen:
            seen.add(cit)
            citations.append(cit)
    return citations


STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def extract_content_tokens(text: str) -> Set[str]:
    """Extracts lowercase alphanumeric content tokens excluding stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]{2,}\b", text.lower())
    return {t for t in tokens if t not in STOP_WORDS}


class ClaimVerifier:
    """
    Verifies claims against candidate evidence chunks.

    IMPORTANT HEURISTIC CAVEATS:
    - By default, ClaimVerifier uses content-token lexical overlap.
    - Token overlap is a surface lexical heuristic, NOT proof of semantic entailment,
      factual accuracy, or logical validity.
    - It CANNOT detect semantic negation, numerical discrepancies, polarity shifts,
      or entity substitutions when surrounding vocabulary is largely identical.
    - Lexical overlap scores are uncalibrated heuristics and do NOT represent confidence
      probabilities, nor do they guarantee zero hallucinations.
    - For true semantic entailment verification, inject a dedicated NLI model
      or verification callable via `verifier_fn`.
    """

    def __init__(
        self,
        support_threshold: float = 0.5,
        partial_threshold: float = 0.25,
        verifier_fn: Optional[Callable[[str, Evidence], SupportStatus]] = None,
    ):
        if not (0.0 <= partial_threshold <= 1.0):
            raise ValueError(f"partial_threshold must be between 0.0 and 1.0, got {partial_threshold}")
        if not (0.0 <= support_threshold <= 1.0):
            raise ValueError(f"support_threshold must be between 0.0 and 1.0, got {support_threshold}")
        if partial_threshold > support_threshold:
            raise ValueError(
                f"partial_threshold ({partial_threshold}) cannot exceed support_threshold ({support_threshold})"
            )
        self.support_threshold = support_threshold
        self.partial_threshold = partial_threshold
        self.verifier_fn = verifier_fn

    def compute_overlap(self, claim_text: str, evidence_text: str) -> Tuple[float, Set[str]]:
        claim_tokens = extract_content_tokens(claim_text)
        evidence_tokens = extract_content_tokens(evidence_text)
        if not claim_tokens:
            if claim_text.strip().lower() in evidence_text.lower():
                return 1.0, set()
            return 0.0, set()
        matched = claim_tokens & evidence_tokens
        ratio = len(matched) / len(claim_tokens)
        return ratio, matched

    def verify(
        self,
        claim_text: str,
        candidate_evidence: List[Evidence],
    ) -> Tuple[SupportStatus, List[Evidence]]:
        """
        Evaluates a claim against candidate evidence chunks.

        Semantics:
        - If candidate_evidence is empty: Returns (NOT_ENOUGH_EVIDENCE, []).
        - If candidate_evidence is non-empty but no chunk satisfies the selection rule:
          Returns (UNSUPPORTED, []).
        - If evidence satisfies the support threshold: Returns (SUPPORTED, supporting_chunks).
        - If evidence satisfies only the partial threshold: Returns (PARTIALLY_SUPPORTED, supporting_chunks).
        """
        if not candidate_evidence:
            return SupportStatus.NOT_ENOUGH_EVIDENCE, []

        if self.verifier_fn is not None:
            supported_chunks: List[Evidence] = []
            partial_chunks: List[Evidence] = []
            for ev in candidate_evidence:
                status = self.verifier_fn(claim_text, ev)
                if status == SupportStatus.SUPPORTED:
                    supported_chunks.append(ev)
                elif status == SupportStatus.PARTIALLY_SUPPORTED:
                    partial_chunks.append(ev)

            if supported_chunks:
                return SupportStatus.SUPPORTED, supported_chunks
            elif partial_chunks:
                return SupportStatus.PARTIALLY_SUPPORTED, partial_chunks
            else:
                return SupportStatus.UNSUPPORTED, []

        scored_evidence: List[Tuple[float, Evidence]] = []
        for ev in candidate_evidence:
            ratio, _ = self.compute_overlap(claim_text, ev.evidence_text)
            if ratio >= self.partial_threshold:
                scored_evidence.append((ratio, ev))

        if not scored_evidence:
            return SupportStatus.UNSUPPORTED, []

        # Sort candidate evidence descending by overlap ratio
        scored_evidence.sort(key=lambda x: x[0], reverse=True)
        max_ratio = scored_evidence[0][0]

        if max_ratio >= self.support_threshold:
            # Enforce selection rule: only return chunks meeting the support threshold
            supporting_chunks = [ev for ratio, ev in scored_evidence if ratio >= self.support_threshold]
            return SupportStatus.SUPPORTED, supporting_chunks
        else:
            # Only return chunks meeting the partial threshold
            supporting_chunks = [ev for ratio, ev in scored_evidence if ratio >= self.partial_threshold]
            return SupportStatus.PARTIALLY_SUPPORTED, supporting_chunks

    def filter_supporting_evidence(
        self,
        text: str,
        candidate_evidence: List[Evidence],
        threshold: Optional[float] = None,
    ) -> List[Evidence]:
        """Filters candidate evidence chunks to those meeting the selection rule."""
        if self.verifier_fn is not None:
            matching: List[Evidence] = []
            for ev in candidate_evidence:
                status = self.verifier_fn(text, ev)
                if status in (SupportStatus.SUPPORTED, SupportStatus.PARTIALLY_SUPPORTED):
                    matching.append(ev)
            return matching

        min_thresh = threshold if threshold is not None else self.partial_threshold
        matching_heur: List[Evidence] = []
        for ev in candidate_evidence:
            ratio, _ = self.compute_overlap(text, ev.evidence_text)
            if ratio >= min_thresh:
                matching_heur.append(ev)
        return matching_heur


@dataclass
class Claim:
    text: str
    evidence: List[Evidence] = field(default_factory=list)
    support_status: SupportStatus = SupportStatus.NOT_ENOUGH_EVIDENCE

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


@dataclass
class CorroboratingFinding:
    finding: str
    supporting_papers: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


@dataclass
class ConflictingFinding:
    topic: str
    papers: List[str] = field(default_factory=list)
    explanation: str = ""
    evidence: List[Evidence] = field(default_factory=list)

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


@dataclass
class ResearchGap:
    finding: str
    relevant_papers: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    def get_citations(self, style: str = "bracket") -> List[str]:
        return build_citations(self.evidence, style=style)


class EvidenceAggregator:
    """
    In-memory service responsible for collecting, deduplicating,
    grouping, and linking Evidence objects across papers and claims.
    """

    def __init__(self, evidence_list: Optional[List[Evidence]] = None):
        self.evidence: List[Evidence] = []
        if evidence_list:
            self.add_evidence(evidence_list)

    def add_evidence(self, evidence: Union[Evidence, List[Evidence]]) -> None:
        if isinstance(evidence, Evidence):
            self.evidence.append(evidence)
        elif isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, Evidence):
                    raise TypeError(f"Expected Evidence object, got {type(item)}")
                self.evidence.append(item)
        else:
            raise TypeError(f"Expected Evidence or List[Evidence], got {type(evidence)}")

    def get_all_evidence(self) -> List[Evidence]:
        return list(self.evidence)

    def deduplicate(self) -> List[Evidence]:
        """
        Deduplicates identical evidence instances based on chunk_id or (document_id, page, content).
        Preserves original order.
        """
        seen: Set[Any] = set()
        deduped: List[Evidence] = []
        for ev in self.evidence:
            if ev.chunk_id:
                key = ("chunk_id", ev.chunk_id)
            elif ev.parent_id:
                key = ("parent_id", ev.parent_id, ev.evidence_text.strip())
            else:
                key = ("content", ev.document_id, ev.page, ev.evidence_text.strip())

            if key not in seen:
                seen.add(key)
                deduped.append(ev)

        self.evidence = deduped
        return deduped

    def group_by_document(self, by: str = "document_name") -> Dict[str, List[Evidence]]:
        """
        Groups aggregated evidence by document name or document id.
        """
        grouped: Dict[str, List[Evidence]] = {}
        for ev in self.evidence:
            key = ev.document_id if by == "document_id" else ev.document_name
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(ev)
        return grouped

    def get_sources_for_paper(self, paper_identifier: str) -> List[Evidence]:
        """
        Retrieves all evidence belonging to a given document_id or document_name.
        """
        return [
            ev for ev in self.evidence
            if ev.document_id == paper_identifier or ev.document_name == paper_identifier
        ]

    def get_sources_for_claim(self, claim: Claim) -> List[Evidence]:
        """
        Retrieves attached evidence for a specific claim.
        """
        if not isinstance(claim, Claim):
            raise TypeError(f"Expected Claim instance, got {type(claim)}")
        return list(claim.evidence)

    def build_source_references(self, style: str = "bracket") -> List[str]:
        """
        Produces sorted, deduplicated citation strings for all aggregated evidence.
        """
        deduped = self.deduplicate()
        return build_citations(deduped, style=style)
