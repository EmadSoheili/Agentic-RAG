# Here we define the graph that will define the flow of our application. 
# The graph will be used to create a state machine that will define the flow of our application.
# Here we will define the nodes and edges of the graph, and how they will be connected to each other.


from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, StateGraph         # To create a state graph that will define the flow of our application.
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH            # To use the constants to avoid duplicate strings when calling nodes in the graph.
from graph.nodes import retrieve, grade_documents, generate, web_search             # Import the nodes that we defined
from graph.state import GraphState           # To define the state of the graph.
from graph.chains.answer_grader import answer_grader            # To use the answer grader chain to grade the documents.
from graph.chains.hallucination_grader import hallucinatin_grader            # To use the hallucination grader chain to grade the documents.
from graph.chains.router import question_router, RouteQuery           # To use the router chain to route the question to the appropriate node.


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


# Define the function that will grade the generation for hallucination and whether it answers the question.
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATION---")

    # Get the information from the state that we need to grade the generation for hallucination.
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    # Invoking the hallucination grader chain
    # Passing the information to the hallucination grader chain to grade the generation for hallucination.
    # The response will be boolean, 'yes' or 'no', as we defined the structure in the hallucination grader chain.
    score = hallucinatin_grader.invoke({"documents": documents, "generation": generation})

    if hallucination_grade := score.binary_score:           # Check for hallucination in the generation.
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})          # Invoking the answer grader chain

        if answer_grade := score.binary_score:          # Check if the generation answers the question.
            print("---DECISION: GENERATION ADDRESSED THE QUESTION---")
            return "useful"

        else:
            print("---DECISION: GENERATION DID NOT ADDRESS THE QUESTION---")
            return "not useful"

    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"


#
def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})           # Invoke the router chain to route the question to the appropriate node.
        # The result will have either 'vectorstore' or 'web_search' under datasource as we defined the structure in the RouteQuery class.

    if source.datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEB_SEARCH
    else:
        print("---ROUTE QUESTION TO VECTORESTORE---")
        return RETRIEVE


# Create a state graph that will define the flow of our application.
workflow = StateGraph(GraphState)

# Create nodes
workflow.add_node(RETRIEVE, retrieve)           # Add the retrieve node to the graph.
workflow.add_node(GRADE_DOCUMENTS, grade_documents)           # Add the grade_documents node to the graph.
workflow.add_node(GENERATE, generate)           # Add the generate node to the graph.
workflow.add_node(WEB_SEARCH, web_search)           # Add the web_search node to the graph.

workflow.set_conditional_entry_point(
    route_question,           # The function that will decide which node to enter first based on the question.
    {
        WEB_SEARCH: WEB_SEARCH,           # If the question is routed to web search, enter the web_search node.
        RETRIEVE: RETRIEVE,           # If the question is routed to vectorstore
    }
)


# Create edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)           # Add an edge from the retrieve node to the grade_documents node.

workflow.add_conditional_edges(
    GRADE_DOCUMENTS,           # From this node to...
    decide_to_generate,           # The condtional function
    {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE}           # To one of these nodes based on the output of the conditional function.
)

workflow.add_conditional_edges(
    GENERATE,           # From this node to...
    grade_generation_grounded_in_documents_and_question,           # The condtional function
    {"useful": END, "not useful": WEB_SEARCH, "not supported": GENERATE}           # To one of these nodes based on the output of the conditional function.
)

workflow.add_edge(WEB_SEARCH, GENERATE)           # Add an edge from the web_search node to the end of the graph.


# Compile and visualize the graph
app = workflow.compile()            # Compile the graph into a runnable application.
app.get_graph().draw_mermaid_png(output_file_path="graph.png")           # Draw the graph and save it as a PNG file.