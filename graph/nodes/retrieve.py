# Here we define the retrieve node in our graph.
# This node will take the current state of the graph, which includes the question, and use a retriever to get relevant documents.
# This retrieval will use the vector database to find documents that are relevant to the question.
# Note that we already have embedded the source documents in the vector database.


from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documnets = retriever.invoke(question)

    return {"documents": documnets, "question": question}           # Write the retreived documents and the question to the state.
