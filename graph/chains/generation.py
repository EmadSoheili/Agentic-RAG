# Here we are defining the generate chain in our graph.
# This chain will take the retrieved documents and the question from the state.
# Then use a language model to generate an answer to the question based on the retrieved documents.


from langchain_core.prompts import ChatPromptTemplate           # To create a prompt template for the LLM.
from langchain_core.output_parsers import StrOutputParser           # To parse the output of the LLM into a string.
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough            # To create a passthrough runnable that will pass the input to the next runnable in the chain.


# Initialize the LLM with the specified model and temperature
MODEL = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0.0)


# Add RAG prompt as a prompt template to the LLM to create a generate chain.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)


# Helper function to extract and format only page text
def format_docs(docs):
    if isinstance(docs, list):
        return "\n\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs)
    return str(docs)


# Chain the prompt, LLM, and output parser together to create a generate chain.
generation_chain = (
    RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
    | prompt
    | llm
    | StrOutputParser()
)