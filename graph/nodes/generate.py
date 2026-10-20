# Here we create the Generate node in our graph.
# This node will take the current state of the graph, which includes the question and the retrieved documents.
# Then use the generate chain to generate an answer to the question based on the retrieved documents.
# Them write the generated answer, the question, and the documents to the state.


from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})

    # Write the generated answer, the question, and the documents to the state.
    return {"documents": documents, "question": question, "generation": generation}