import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline

from typing import Any, Optional
from retrieval.retrieve_llm_pipeline import get_llm
from retrieval.retriver import Retriever


class BasicRetriever:
    def __init__(self, retriever: Any, llm: Optional[Any] = None):
        self.retriever = retriever
        self.llm = llm if llm is not None else get_llm()

        self.prompt = ChatPromptTemplate.from_template("""You are an expert research and technical document evidence assistant.

Answer the user's question using ONLY the provided context.

Rules:
- Answer ONLY what the user asked based on the provided context.
- Do not add unrelated information from the context.
- Do not invent or infer information that is not explicitly supported by the context.
- Keep the answer concise, accurate, and direct.
- If the context does not contain enough information to answer the question, say:
  "The provided documents do not contain enough information to answer this."

Context:
{context}

Question:
{question}
""")

        # prompt → LLM
        self.chain = self.prompt | self.llm


    def generate(self, query: str) -> dict:
        results = self.retriever.retrieve(query)

        # Clean abstention: If no documents were retrieved, do NOT invoke the LLM.
        if not results:
            return {
                "answer": "The provided documents do not contain enough information to answer this.",
                "citations": [],
            }

        # Keep metadata out of the prompt so the model receives only source text.
        context = "\n\n".join(
            result.page_content
            for result in results
        )

        response = self.chain.invoke({
            "context": context,
            "question": query
        })

        citations = []
        for result in results:
            raw_page = result.metadata.get("page", 0)
            try:
                page_num = int(raw_page) + 1
            except (ValueError, TypeError):
                page_num = 1
            citations.append({
                "document_id": result.metadata.get("document_id"),
                "document_name": result.metadata.get("document_name"),
                "source": result.metadata.get("source"),
                "title": result.metadata.get("title"),
                "page": page_num,
                "chunk_id": result.metadata.get("chunk_id"),
                "parent_id": result.metadata.get("parent_id"),
            })

        return {
            "answer": response,
            "citations": citations,
        }

