# Here we define the state of our graph,
# which includes the question, the generation from the LLM, whether to add web search, and the list of documents.
# This GraphState is like the final status of our graph at each step.
# This class will work like a dictionary, but with specific keys and types for each key.
# At each step, nodes might overwrite the values of these keys, and we will use this state to keep track of the graph's progress.


from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]