# Here we are creating the Grade Document node.
# This node will decide received documents are relevant and should be used or filtered out.


from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


# Define the grade_documents function, which takes the current state of the graph as input
# This function will return a dictionary containing the filtered documents and the updated web_search state.
def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """


