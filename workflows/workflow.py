from pathlib import Path
from typing import Optional, TypedDict
from langgraph.graph import END, START, StateGraph

class RAGState(TypedDict):
    question: str
    documents: list
    answer: str
    confidence: str
    citations: list
    retry_count: int

def retrieve_node(state: RAGState):
    print("Retrieving...")
    
    return {
        "documents": ["dummy document"]
    }


def generate_node(state: RAGState):
    print("Generating...")
    
    return {
        "answer": "dummy answer"
    }

def retrieve_more_node(state: RAGState):
    print("Retrieving more evidence...")

    return {
        "retry_count": state["retry_count"] + 1,
        "documents": ["additional dummy document"]
    }

def evidence_check_node(state: RAGState):
    if len(state["documents"]) > 0:
        return {"confidence": "HIGH"}
    else:
        return {"confidence": "INSUFFICIENT"}

def route_after_evidence(state: RAGState):
    if state["confidence"] == "HIGH":
        return "generate"
    else:
        return "retrieve_more"

    

graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evidence_check", evidence_check_node)
graph.add_node("generate", generate_node)
graph.add_node("retrieve_more", retrieve_more_node)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "evidence_check")

graph.add_conditional_edges(
    "evidence_check",
    route_after_evidence,
    {
        "generate": "generate",
        "retrieve_more": "retrieve_more"
    }
)
graph.add_edge("retrieve_more", "evidence_check")
graph.add_edge("generate", END)


app = graph.compile()


def export_workflow_diagram(output_path: Optional[Path] = None) -> Path:
    target_path = output_path or Path(__file__).with_name("rag_workflow.png")
    app.get_graph().draw_mermaid_png(output_file_path=str(target_path))
    return target_path


def run_rag_workflow(question: str):
    state: RAGState = {
        "question": question,
        "documents": ["Syllabus in unit 2 is about the study of data structures and algorithms, including topics such as arrays, linked lists, stacks, queues, trees, graphs, sorting and searching algorithms, and algorithmic complexity analysis. The syllabus also covers practical applications and problem-solving techniques related to these concepts."],
        "answer": "",
        "confidence": "",
        "citations": [],
        "retry_count": 0
    }
    
    result = app.invoke(state)
    return result


if __name__ == "__main__":
    diagram_path = export_workflow_diagram()
    print(f"Graph diagram saved to {diagram_path}")
    sample_res = run_rag_workflow("What is in unit 2?")
    print("Workflow execution result:\n", sample_res)

