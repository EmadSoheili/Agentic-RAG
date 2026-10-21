# Here we create a chain that will check if the answer is grounded in the retrieved documents.
# In other words it will check if the answer is hallucinated or not.


from langchain_core.prompts import ChatPromptTemplate           # To create a prompt template for the LLM.
from pydantic import BaseModel, Field                           # To create a Pydantic model for the output of the LLM.
from langchain_core.runnables import RunnableSequence          # To create a sequence of runnables that will be executed in order.
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize the LLM with the specified model and temperature
MODEL = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)


# Define a class to represent the output of the hallucination grader.
# This class will be used to parse the output of the LLM and check if the answer is grounded in the retrieved documents.
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in the generated answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'?"
    )


# Applying structured output to the LLM to create a hallucination grader.
# Here we use the GradeHallucinations class and tell the llm to follow the structure of the class when generating the output.
structered_llm_grader = llm.with_structured_output(GradeHallucinations)


# Define the system message.
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


# Create a prompt template for the hallucination grader.
# It will take the retrieved documents and the LLM generation as input and craft the prompt.
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
    ]
)


# Create the hallucination grader chain
hallucinatin_grader: RunnableSequence = hallucination_prompt | structered_llm_grader