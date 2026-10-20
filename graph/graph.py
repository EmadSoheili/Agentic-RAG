# 


from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph         # To create a state graph that will define the flow of our application.
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH            # To use the constants to avoid duplicate strings when calling nodes in the graph.
from graph.nodes import retrieve, grade_documents, generate, web_search             # Import the nodes that we defined
from graph.state import GraphState           # To define the state of the graph.


# Define the function that will decide whether to generate an answer or search the web based on the graded documents.
# This function will be used as a node in the graph.
def decide_to_generate(state):
    print("---ASSESS GRADEDE DOCUMENTS---")

    if state["web_search"]:
        print("---DECISION: WEB SEARCH (NOT ALL DOCUMENTS RELEVANT TO THE QUESTION)---")
        return WEB_SEARCH
    else:
        print("---DECISION: GENERATE (ALL DOCUMENTS RELEVANT TO THE QUESTION)---")
        return GENERATE


# Create a state graph that will define the flow of our application.
workflow = StateGraph(GraphState)

# Create nodes
workflow.add_node(RETRIEVE, retrieve)           # Add the retrieve node to the graph.
workflow.add_node(GRADE_DOCUMENTS, grade_documents)           # Add the grade_documents node to the graph.
workflow.add_node(GENERATE, generate)           # Add the generate node to the graph.
workflow.add_node(WEB_SEARCH, web_search)           # Add the web_search node to the graph.
workflow.set_entry_point(RETRIEVE)           # Set the entry point of the graph to the retrieve node.


# Create edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)           # Add an edge from the retrieve node to the grade_documents node.

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,           # From this node to...
    decide_to_generate,           # The condtional function
    {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE}           # To one of these nodes based on the output of the conditional function.
)

workflow.add_edge(WEB_SEARCH, GENERATE)           # Add an edge from the web_search node to the end of the graph.
workflow.add_edge(GENERATE, END)           # Add an edge from the generate node to the end of the graph.


app = workflow.compile()            # Compile the graph into a runnable application.
app.get_graph().draw_mermaid_png(output_file_path="graph.png")           # Draw the graph and save it as a PNG file.