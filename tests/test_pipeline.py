import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM
from retrieval.pipeline import BasicRetriever


class MockRetriever:
    def retrieve(self, query):
        return [
            Document(
                page_content="Unit 2 covers data structures including arrays, linked lists, and trees.",
                metadata={
                    "document_id": "doc_1_syllabus",
                    "document_name": "syllabus.pdf",
                    "source": "syllabus.pdf",
                    "title": "Course Syllabus",
                    "page": 1,
                },
            )
        ]


def test_basic_retriever_pipeline():
    mock_retriever = MockRetriever()
    fake_llm = FakeListLLM(responses=["Data structures including arrays, linked lists, and trees."])
    pipeline = BasicRetriever(retriever=mock_retriever, llm=fake_llm)

    result = pipeline.generate("What is syllabus in the unit 2?")
    assert "answer" in result
    assert "citations" in result
    assert result["answer"] == "Data structures including arrays, linked lists, and trees."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["document_id"] == "doc_1_syllabus"
    assert result["citations"][0]["page"] == 2  # 1-indexed display page


def test_basic_retriever_empty_retrieval_abstains():
    class EmptyRetriever:
        def retrieve(self, query):
            return []

    # If the LLM is invoked, FakeListLLM will raise IndexError because responses is empty!
    fake_llm = FakeListLLM(responses=[])
    pipeline = BasicRetriever(retriever=EmptyRetriever(), llm=fake_llm)

    result = pipeline.generate("What is the quantum state vector?")
    assert "answer" in result
    assert result["answer"] == "The provided documents do not contain enough information to answer this."
    assert result["citations"] == []


def test_basic_retriever_metadata_edge_cases():
    class EdgeCaseRetriever:
        def retrieve(self, query):
            return [
                Document(
                    page_content="Content with invalid page metadata.",
                    metadata={
                        "document_id": "doc_edge",
                        "document_name": "edge.pdf",
                        "page": "invalid_page_number",
                        "chunk_id": "c_edge_1",
                        "parent_id": "p_edge_1",
                    },
                )
            ]

    fake_llm = FakeListLLM(responses=["Summary of content."])
    pipeline = BasicRetriever(retriever=EdgeCaseRetriever(), llm=fake_llm)

    result = pipeline.generate("Query")
    assert len(result["citations"]) == 1
    cit = result["citations"][0]
    assert cit["page"] == 1  # Fallback to page 1
    assert cit["chunk_id"] == "c_edge_1"
    assert cit["parent_id"] == "p_edge_1"



if __name__ == "__main__":
    from retrieval.retriver import Retriever

    answerer = BasicRetriever(Retriever())
    res = answerer.generate("What is syllabus in the unit 2?")
    print("Answer:\n", res["answer"])
    print("\nCitations:")
    for citation in res["citations"]:
        print(citation)