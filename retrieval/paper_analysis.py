import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from retrieval.evidence import Claim, ClaimVerifier, Evidence, SupportStatus
from retrieval.retriver import Retriever


@dataclass
class PaperAnalysis:
    document_id: str
    document_name: str
    title: str
    research_problem: str
    methodology: str
    major_claims: str
    evidence: str
    limitations: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    source_evidence: List[Evidence] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    diagnostics: Optional[Dict[str, Any]] = None


PAPER_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""You are an expert research paper evidence analyst.

Analyze the provided research paper excerpt and produce a structured analysis of this paper ONLY.

Rules:
- Analyze ONLY the current paper's provided evidence.
- Do not use information from other papers.
- Do not invent missing details.
- If a requested field cannot be supported by the evidence, explicitly say that the information is not available in the provided evidence.
- Keep the analysis concise.
- Preserve factual distinctions between what the paper claims and what its experiments actually demonstrate.
- Do not perform cross-paper comparisons.
- Do not generate generic filler.

Format your output EXACTLY with the following headers:
TITLE: <paper title or identification>
RESEARCH PROBLEM: <core problem or objective addressed>
METHODOLOGY: <approach, model, or methodology proposed>
MAJOR CLAIMS: <main claims made by the authors>
EVIDENCE: <experimental support, benchmark data, or empirical findings>
LIMITATIONS: <limitations stated in the paper or "Not available in the provided evidence.">

Provided Evidence for {document_name}:
{context}

Structured Paper Analysis:
""")


class PaperAnalyzer:
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[Any] = None,
        verifier: Optional[ClaimVerifier] = None,
    ):
        self.retriever = retriever
        if llm is None:
            from retrieval.retrieve_llm_pipeline import llm as default_llm
            self.llm = default_llm
        else:
            self.llm = llm

        self.verifier = verifier if verifier is not None else ClaimVerifier()
        self.chain = PAPER_ANALYSIS_PROMPT | self.llm

    def _ensure_retriever(self, file_paths: Optional[Union[str, List[str]]] = None) -> Retriever:
        if self.retriever is None:
            self.retriever = Retriever(file_paths=file_paths, rebuild_index=bool(file_paths))
        return self.retriever

    def analyze_paper(
        self,
        document_id: str,
        document_name: Optional[str] = None,
        k: int = 4,
    ) -> PaperAnalysis:
        retriever = self._ensure_retriever()

        # Query relevant sections for this specific paper
        query = "title research problem objective methodology approach major claims experimental evidence limitations"
        if hasattr(retriever, "retrieve_with_diagnostics"):
            evidence_docs, diagnostics = retriever.retrieve_with_diagnostics(query=query, k=k, document_id=document_id)
        else:
            evidence_docs = retriever.retrieve(query=query, k=k, document_id=document_id)
            diagnostics = None

        # Fallback to direct parent documents if retrieval returned empty
        if not evidence_docs and hasattr(retriever, "parents") and retriever.parents:
            evidence_docs = [
                parent for parent in retriever.parents
                if parent.metadata.get("document_id") == document_id
            ][:k]

        if not document_name and evidence_docs:
            document_name = evidence_docs[0].metadata.get("document_name", "Unknown Document")
        elif not document_name:
            document_name = "Unknown Document"

        # Clean abstention: If no evidence could be retrieved or found, do not invoke LLM.
        if not evidence_docs:
            return PaperAnalysis(
                document_id=document_id,
                document_name=document_name,
                title=document_name,
                research_problem="Not available in the provided evidence.",
                methodology="Not available in the provided evidence.",
                major_claims="Not available in the provided evidence.",
                evidence="Not available in the provided evidence.",
                limitations="Not available in the provided evidence.",
                citations=[],
                source_evidence=[],
                claims=[],
                diagnostics=diagnostics,
            )

        # Separate metadata and build text context for LLM
        context = "\n\n".join(doc.page_content for doc in evidence_docs)

        llm_response = self.chain.invoke({
            "document_name": document_name,
            "context": context,
        })

        parsed_fields = self._parse_analysis_output(llm_response, default_title=document_name)

        # Preserve retrieved documents deterministically as Evidence objects
        source_evidence = [Evidence.from_document(doc) for doc in evidence_docs]

        # Structure individual claims and verify support against source evidence
        raw_claims = parsed_fields["major_claims"].split("\n")
        claim_texts = [
            re.sub(r"^[\s*\-\d\.\)]+", "", line).strip()
            for line in raw_claims
            if line.strip()
        ]
        valid_claim_texts = [
            ct for ct in claim_texts
            if ct and "not available in the provided evidence" not in ct.lower()
        ]

        claims: List[Claim] = []
        for ct in valid_claim_texts:
            status, supporting_ev = self.verifier.verify(ct, source_evidence)
            claims.append(
                Claim(
                    text=ct,
                    evidence=supporting_ev,
                    support_status=status,
                )
            )

        # Preserve original document metadata deterministically as citations
        citations = []
        for doc in evidence_docs:
            raw_page = doc.metadata.get("page", 0)
            try:
                page_num = int(raw_page) + 1
            except (ValueError, TypeError):
                page_num = 1
            citations.append({
                "document_id": doc.metadata.get("document_id", document_id),
                "document_name": doc.metadata.get("document_name", document_name),
                "source": doc.metadata.get("source"),
                "page": page_num,
            })

        return PaperAnalysis(
            document_id=document_id,
            document_name=document_name,
            title=parsed_fields["title"],
            research_problem=parsed_fields["research_problem"],
            methodology=parsed_fields["methodology"],
            major_claims=parsed_fields["major_claims"],
            evidence=parsed_fields["evidence"],
            limitations=parsed_fields["limitations"],
            citations=citations,
            source_evidence=source_evidence,
            claims=claims,
            diagnostics=diagnostics,
        )

    def analyze_papers(
        self,
        file_paths: Optional[Union[str, List[str]]] = None,
        k: int = 4,
    ) -> List[PaperAnalysis]:
        retriever = self._ensure_retriever(file_paths)
        indexed_papers = retriever.get_indexed_papers()

        analyses: List[PaperAnalysis] = []
        for paper_meta in indexed_papers:
            doc_id = paper_meta["document_id"]
            doc_name = paper_meta["document_name"]
            analysis = self.analyze_paper(document_id=doc_id, document_name=doc_name, k=k)
            analyses.append(analysis)

        return analyses

    @staticmethod
    def _parse_analysis_output(raw_text: str, default_title: str) -> Dict[str, str]:
        """Extracts structured fields from the LLM output with fallback defaults."""
        fields = {
            "title": default_title,
            "research_problem": "Not available in the provided evidence.",
            "methodology": "Not available in the provided evidence.",
            "major_claims": "Not available in the provided evidence.",
            "evidence": "Not available in the provided evidence.",
            "limitations": "Not available in the provided evidence.",
        }

        header_patterns = [
            ("title", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:TITLE)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:RESEARCH\s*PROBLEM|METHODOLOGY|MAJOR|EVIDEN|LIMITATION)|$)"),
            ("research_problem", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:RESEARCH\s*PROBLEM|PROBLEM|OBJECTIVE)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:METHODOLOGY|APPROACH|MAJOR|EVIDEN|LIMITATION)|$)"),
            ("methodology", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:METHODOLOGY|APPROACH)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:MAJOR|CLAIMS|EVIDEN|LIMITATION)|$)"),
            ("major_claims", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:MAJOR\s*\w*\s*CLAIMS|KEY\s*CLAIMS|CLAIMS)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:EVIDEN|LIMITATION)|$)"),
            ("evidence", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:EVIDENCE|EVIDENTIARY\s*SUPPORT|EXPERIMENTAL\s*(?:SUPPORT|RESULTS))\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:LIMITATION)|$)"),
            ("limitations", r"(?:^|\n)\s*[\#\*\-\d\.\s]*(?:LIMITATIONS?)\s*[\#\*\-\s]*:\s*(.*?)(?=\n\s*[\#\*\-\d\.\s]*(?:Assistant:|###)|$)"),
        ]

        for key, pattern in header_patterns:
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                val = match.group(1).strip().strip("*#-_ \t\n")
                if val:
                    fields[key] = val

        return fields


def analyze_papers(
    file_paths: Optional[Union[str, List[str]]] = None,
    retriever: Optional[Retriever] = None,
    llm: Optional[Any] = None,
    verifier: Optional[ClaimVerifier] = None,
    k: int = 4,
) -> List[PaperAnalysis]:
    analyzer = PaperAnalyzer(retriever=retriever, llm=llm, verifier=verifier)
    return analyzer.analyze_papers(file_paths=file_paths, k=k)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run single or multi-paper analysis on research papers.")
    parser.add_argument("pdf_paths", nargs="+", help="Path(s) to the PDF paper(s) to analyze.")
    parser.add_argument("--k", type=int, default=4, help="Number of evidence chunks to retrieve per paper.")
    parser.add_argument("--db-dir", type=str, default=None, help="Chroma persistence directory.")
    args = parser.parse_args()

    paths = args.pdf_paths
    print(f"Starting paper analysis on {len(paths)} paper(s):")
    for p in paths:
        print(f" - {p}")

    analyses = analyze_papers(file_paths=paths, k=args.k)

    for idx, analysis in enumerate(analyses, 1):
        print(f"\n========================================================")
        print(f"Paper Analysis [{idx}/{len(analyses)}]: {analysis.document_name}")
        print(f"Document ID:      {analysis.document_id}")
        print(f"Title:            {analysis.title}")
        print(f"Research Problem: {analysis.research_problem}")
        print(f"Methodology:      {analysis.methodology}")
        print(f"Major Claims:     {analysis.major_claims}")
        print(f"Evidence:         {analysis.evidence}")
        print(f"Limitations:      {analysis.limitations}")
        print(f"\nCitations ({len(analysis.citations)}):")
        for cit in analysis.citations:
            print(f"  - [{cit.get('document_name')}] Page {cit.get('page')}: {cit.get('source')}")
        print(f"========================================================\n")
