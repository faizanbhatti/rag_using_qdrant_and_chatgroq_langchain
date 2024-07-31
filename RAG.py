import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

# Load environment variables from a .env file
load_dotenv()

# Get the API key for Groq and host information for the vector database from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
vector_db_host = os.getenv("VECTOR_DB_HOST")
vector_db_post = os.getenv("VECTOR_DB_PORT")
url = f"{vector_db_host}:{str(vector_db_post)}"  # Construct the URL for the vector database

# Initialize embeddings using OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Initialize sparse embeddings using FastEmbedSparse with a specified model
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


def document_loader(path):
    """
    Load and preprocess PDF documents.

    Args:
        path (str): Path to the PDF file.

    Returns:
        list: A list of split and preprocessed text segments.
    """
    # Load documents from the given PDF file path
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split the documents into chunks of size 500 with an overlap of 50 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Preprocess the text: convert to lowercase and strip whitespace
    for text in texts:
        text.page_content = text.page_content.lower().strip()

    return texts


def data_ingestion_to_vector_db(
    documents,
    embeddings,
    sparse_embeddings,
    url,
    collection_name="gpt_db",
):
    """
    Ingest data into the vector database (Qdrant).

    Args:
        documents (list): List of preprocessed documents.
        embeddings (object): Embeddings instance for dense embeddings.
        sparse_embeddings (object): Instance for sparse embeddings.
        url (str): URL of the Qdrant server.
        collection_name (str): Name of the collection to store data.

    Returns:
        QdrantVectorStore: Instance of the Qdrant vector store.
    """
    # Create a Qdrant vector store and ingest documents
    qdrant = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,  # Use hybrid retrieval mode (dense + sparse)
        prefer_grpc=False,  # Set to use HTTP instead of gRPC
        force_recreate=True,  # Force recreate the index if it exists
        url=url,
    )
    print("Qdrant Index Created.......")
    return qdrant


def search_from_vector_db(
    embeddings,
    sparse_embeddings,
    url,
    collection_name="gpt_db",
):
    """
    Search for relevant documents in the vector database.

    Args:
        embeddings (object): Embeddings instance for dense embeddings.
        sparse_embeddings (object): Instance for sparse embeddings.
        url (str): URL of the Qdrant server.
        collection_name (str): Name of the collection to search in.

    Returns:
        QdrantVectorStore: Instance of the Qdrant vector store.
    """
    # Retrieve the vector store from an existing collection
    vectordb = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        collection_name=collection_name,
        url=url,
        retrieval_mode=RetrievalMode.HYBRID,  # Use hybrid retrieval mode
    )
    return vectordb


# Load documents from a sample PDF file
docs = document_loader("sample_data.pdf")

# Ingest the documents into the vector database
data_ingestion_to_vector_db(docs, embeddings, sparse_embeddings, url)

# Search the vector database and retrieve the vector store
vector_db = search_from_vector_db(embeddings, sparse_embeddings, url)


def invoke(question):
    """
    Perform a similarity search in the vector database with a given question.

    Args:
        question (str): The question to search for.

    Returns:
        list: List of documents that are similar to the question.
    """
    found_docs = vector_db.similarity_search(question)  # Perform similarity search
    return found_docs


# Define a prompt template for generating answers
prompt = PromptTemplate(
    template="""system You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise user
    Question: {question} 
    Context: {context} 
    Answer: assistant""",
    input_variables=["question", "document"],
)

# Initialize a ChatGroq instance with a specified model and temperature
chat = ChatGroq(model="llama3-8b-8192", temperature=0, groq_api_key=groq_api_key)


def format_docs(docs):
    """
    Format the documents into a single string.

    Args:
        docs (list): List of document objects.

    Returns:
        str: Formatted string of document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


# Define a chain of actions: prompt creation, chat completion, and output parsing
rag_chain = prompt | chat | StrOutputParser()

# Run the system with a sample question
question = "What are the capabilities of ChatGPT-4?"

# Invoke the vector database to retrieve relevant documents
docs = invoke(question)

# Measure the time taken for the generation
start_time = time.time()
generation = rag_chain.invoke(
    {"context": docs, "question": question}
)  # Generate the answer
print(generation)  # Print the generated answer
end_time = time.time()

# Calculate and print the time taken for the generation process
time_taken = end_time - start_time
print("Time:", time_taken, "seconds")
