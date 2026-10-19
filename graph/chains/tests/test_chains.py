from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever


# --------------------------------------------------------------------------------------------------
# 1st Test Case: Test the retrieval grader with a question that has a relevant document
# --------------------------------------------------------------------------------------------------
def test_retrieval_grader_answer_yes() -> None:
    # Test case where the answer is "yes"

    question = "agent memory"           # Question: We expect "agent memory" to be relevant to the documents in the vector database.
    docs = retriever.invoke(question)            # Retrieve documents using the retriever
    doc_text = docs[1].page_content         # Assuming the second document is relevant

    # Invoke the retrieval grader with the document and question
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )           

    assert res.binary_score == "yes"           # Assert that the binary score is "yes"



# --------------------------------------------------------------------------------------------------
# 2nd Test Case: Test the retrieval grader with a question that has no relevant document
# --------------------------------------------------------------------------------------------------
def test_retrieval_grader_answer_no() -> None:
    # Test case where the answer is "no"

    question = "agent memory"           # Question: We expect "agent memory" to be relevant to the documents in the vector database.
    docs = retriever.invoke(question)            # Retrieve documents using the retriever
    doc_text = docs[1].page_content         # Assuming the second document is relevant

    # Invoke the retrieval grader with the document and question
    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "How to make a Pizza?", "document": doc_text}          # Use an unrelated question to get No as the answer
    )

    assert res.binary_score == "no"           # Assert that the binary score is "no"


