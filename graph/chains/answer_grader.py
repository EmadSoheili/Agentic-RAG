# 


from langchain_core.prompts import ChatPromptTemplate           # To create a prompt template for the LLM.
from pydantic import BaseModel, Field                           # To create a Pydantic model for the output of the LLM.
from langchain_core.runnables import RunnableSequence          # To create a sequence of runnables that will be executed in order.
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize the LLM with the specified model and temperature
MODEL = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)


# Define a class to represent the output of the answer grader.
# This class will be used to parse the output of the LLM and check if the answer is relevant to the question.
class GradeAnswers(BaseModel):
    """Binary score for answer relevance to the question."""

    binary_score: str = Field(
        description="Answer adresses the question, 'yes' or 'no'?"
    )


# Applying structured output to the LLM to create a answer grader.
# Here we use the GradeAnswers class and tell the llm to follow the structure of the class when generating the output.
structured_llm_grader = llm.with_structured_output(GradeAnswers)


# Define the system message.
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""


# Create a prompt template for the answer grader.
# It will take the retrieved documents and the LLM generation as input and craft the prompt.
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
    ]
)


# Create the answer grader chain
answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

