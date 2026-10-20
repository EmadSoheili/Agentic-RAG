# In this project, we won't focus on ingestion optimization and just go ahead with default ingestion.
# Focus of this project is retrival part. So the ingestion part will be kept simple and straightforward.

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_text_splitters import RecursiveCharacterTextSplitter         # To split the text into chunks
from langchain_unstructured import UnstructuredLoader           # To load the text from the file
from langchain_chroma import Chroma           # To create the vector store
from langchain_google_genai import GoogleGenerativeAIEmbeddings         #Google Gemini embedding


# Sources: URLs to ingest
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Define the embedding model to use for creating embeddings
EMBEDDING = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


# Load the text from the URLs
docs = [UnstructuredLoader(web_url=url, chunking_strategy="basic", max_characters=1000000).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]  # Flatten the list of lists


# Split the text into chunks (chunk_size=250, chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)           
doc_splits = text_splitter.split_documents(docs_list)


# Clean heavy and base64 metadata fields produced by UnstructuredLoader
for doc in doc_splits:
    doc.metadata.pop("orig_elements", None)
    doc.metadata.pop("emphasized_text_contents", None)
    doc.metadata.pop("link_texts", None)


# Create a vector store from the document chunks using Google Gemini embeddings and Chroma as the vector store
# It is commented out to avoid re-ingesting the data every time the script is run. 
# Uncomment it if you want to ingest the data again.
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=EMBEDDING,
#     persist_directory="./.chroma"
# )


#
retriever = Chroma(
    collection_name="rag-chroma",
    embedding_function=EMBEDDING,
    persist_directory="./.chroma"
).as_retriever()


# print ("Done")