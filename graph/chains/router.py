# Here 


from typing import Literal
    # Literal is to specify that a variable can only take on some predefined values.
    # This is for type checking and validation purposes.

from langchain_core.prompts import ChatPromptTemplate           # To create a prompt template for the LLM.
from pydantic import BaseModel, Field                           # To create a Pydantic model for the output of the LLM.
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize the LLM with the specified model and temperature
MODEL = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)


class RouteQuery(BaseModel):
    """Route a user query to the most relevant data source."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,            # The ellipsis indicates that this field is required and must be provided.
        description="Given a user question choose to route it to web search or vectorstore.",
    )


# Create structured output for the LLM
# Basically we are binding this pydantic class to be called as a function calling the llm.
structured_llm_router = llm.with_structured_output(RouteQuery)


# Define the system message for the LLM
system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""


# Create a prompt template for the router
router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question}"),
    ]
)


# Create the question_router chain
question_router = router_prompt | structured_llm_router