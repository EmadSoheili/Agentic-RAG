# Here we create the WebSearch node in our graph.
# This node will take the current state of the graph, which includes the question.
# Then includes Tavily to search the web for relevant documents.


from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict
from langchain_core.documents import Document           # To convert Tavily search results into LangChain Document objects.
from langchain_tavily import TavilySearch           # To use Tavily to search the web for relevant documents.
from graph.state import GraphState              # To access the current state of the graph.


web_search_tool = TavilySearch(max_results=3)           # Create an instance of the TavilySearch class.


# 
def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]           # Get the question from the current state of the graph.
    documents = state["documents"] if "documents" in state else None           # Get the documents from the current state of the graph.
        # Since we are getting the document after checking for relevance, so we should only get relevant documents here.

    tavily_results = web_search_tool.invoke({"query": question})["results"]       # Use Tavily to search the web for our question.

    # Convert the Tavily search results into one LangChain Document objects.
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )

    web_results = Document(page_content=joined_tavily_result)           # Convert Search results into a LangChain Document.

    if documents is not None:
        documents.append(web_results)           # Append the web search results to the existing documents.
    else:
        documents = [web_results]           # If there were no existing documents, create a new list with the web search results.

    return {"documents": documents, "question": question}           # Update the state with the new documents and the question.


if __name__ == "__main__":
    web_search(state={"question": "Agent memory", "documents": None})