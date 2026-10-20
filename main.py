from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from graph.graph import app  # Import the compiled graph application


# This is to supress Python warnings and logs that are not relevant to the user.
# -----------------------------------------------------------------------------------
import logging
import warnings

# Filter out the specific Gemini fixed sampling temperature warning
warnings.filterwarnings("ignore", message=".*uses fixed sampling defaults.*")

# Suppress HTTP request and SDK INFO logs
logging.disable(logging.INFO)
# -----------------------------------------------------------------------------------


if __name__ == "__main__":
    print ("Hello, Advanced RAG Agent!")
    print(app.invoke(input={"question": "What is agent memory?"}))          # Invoke the graph application with a sample question