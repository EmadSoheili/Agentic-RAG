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

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # When we found a relevant document, we will append it to the filtered_docs list.
    filtered_docs = []

    # Initialize the web_search flag to False.
    web_search = False

    for d in documents:
        # Use the retrieval grader to get a binary score for the relevance of the documents.
        score = retrieval_grader.invoke({"document": d.page_content, "question": question})

        grade = score.binary_score          # Get the binary score we defined in the GradeDocuments

        if grade.lower() == "yes":          # Check the lower case of binary score
            print("---GRADE: DOCUMENT IS RELEVANT---")
        else:
            print("---GRADE: DOCUMENT IS NOT RELEVANT---")
            web_search = True
            continue            # in this case we won't add the document to the filtered_docs list.

    return {"documents": filtered_docs, "question": question, "web_search": web_search}