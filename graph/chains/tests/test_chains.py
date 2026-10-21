from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from ingestion import retriever
from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import hallucinatin_grader, GradeHallucinations
from graph.chains.router import question_router, RouteQuery



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



# --------------------------------------------------------------------------------------------------
# 3rd Test Case: Test the generation chain with a question that has relevant documents to see if it generates a non-empty string answer.
# --------------------------------------------------------------------------------------------------
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)            # Retrieve documents using the retriever
    generation = generation_chain.invoke({"context": docs, "question": question})           # Invoke the generation chain
    
    assert isinstance(generation, str) and len(generation.strip()) > 0          # Assert that the generation is a non-empty string
        # For seeing the result, you can check LangSmith tracing.



# --------------------------------------------------------------------------------------------------
# 4th Test Case: Test hallucination grader with a relevant question (non-hallucinated answer)
# --------------------------------------------------------------------------------------------------
def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)            # Retrieve documents using the retriever
    
    generation = generation_chain.invoke({"context": docs, "question": question})           # Invoke the generation chain

    res: GradeHallucinations = hallucinatin_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert res.binary_score



# --------------------------------------------------------------------------------------------------
# 5th Test Case: Test hallucination grader with a irrelevant generated answer (hallucinated answer)
# --------------------------------------------------------------------------------------------------
def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)            # Retrieve documents using the retriever
    
    generation = generation_chain.invoke({"context": docs, "question": question})           # Invoke the generation chain

    res: GradeHallucinations = hallucinatin_grader.invoke(
        {"documents": docs,
         "generation": "In order to make pizza we need to first start with the dough."
        }
    )

    assert not res.binary_score



# --------------------------------------------------------------------------------------------------
# 6th Test Case: Test the router chain for getting vectorestore as the routing option.
# --------------------------------------------------------------------------------------------------
def test_router_to_vectorestore() -> None:
    question = "agent memory"
    
    res: RouteQuery = question_router.invoke({"question": question})           # Invoke the router chain

    assert res.datasource == "vectorestore"           # Assert that the router routes to vectorestore



# --------------------------------------------------------------------------------------------------
# 7th Test Case: Test the router chain for getting web_search as the routing option.
# --------------------------------------------------------------------------------------------------
def test_router_to_web_search() -> None:
    question = "How to make a pizza?"
    
    res: RouteQuery = question_router.invoke({"question": question})           # Invoke the router chain

    assert res.datasource == "web_search"           # Assert that the router routes to web_search