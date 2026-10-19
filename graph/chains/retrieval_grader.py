# Here we are creating the Document Grader chain.
# This chain will take the retrieved documents and the question from the state.
# Then use a language model to grade the relevance of the documents to the question.
# We are chaining a prompt template with a structured output LLM to create a retrieval grader.


from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize the LLM with the specified model and temperature
MODEL = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)


# Define the class for grading the relevance of retrieved documents
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'."
    )


# To force the LLM to output a structured response that we defined in the GradeDocuments class.
structured_llm_grader = llm.with_structured_output(GradeDocuments)


# Define system message
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""


# Define the prompt template for grading the relevance of retrieved documents
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


# Chain the prompt and the structured output LLM together to create a retrieval grader.
retrieval_grader = grade_prompt | structured_llm_grader