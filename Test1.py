import os
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
# from langchain_groq import ChatGroq # Commented out for a truly offline setup
from langchain_ollama import ChatOllama # Updated import for local LLM (install with `pip install -U langchain-ollama`)
from langchain_core.documents import Document

# --- 1. Groq API Key (No longer needed for truly offline LLM, but FastEmbed/Chroma setup remains) ---
# If you were to use ChatGroq, this would be needed. For a fully offline system with Ollama, it's not.
# os.environ["GROQ_API_KEY"] = "gsk_PmlVI0plQnBDHatDb7TlWGdyb3FYqnDooV3JVkTePOpINkUPmyX"

# --- 2. Define your Embedding Model (BGE-Large with FastEmbed) ---
# FastEmbed downloads the model to your local machine, enabling offline embeddings.
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# --- 3. Create a Vector Store (ChromaDB) ---
raw_docs = [
    {
        "page_content": "The movie 'Interstellar' directed by Christopher Nolan was released in 2014 and is a science fiction film with a rating of 8.6. It explores themes of space travel and time dilation.",
        "metadata": {"year": 2014, "director": "Christopher Nolan", "genre": "science fiction", "rating": 8.6},
    },
    {
        "page_content": "The film 'Inception' also directed by Christopher Nolan is a mind-bending thriller from 2010 with a rating = 8.8. It involves dream espionage.",
        "metadata": {"year": 2010, "director": "Christopher Nolan", "genre": "thriller", "rating": 8.8},
    },
    {
        "page_content": "Pulp Fiction, a crime classic by Quentin Tarantino, came out in 1994 and has a rating = 8.9. It features interconnected stories in Los Angeles underworld.",
        "metadata": {"year": 1994, "director": "Quentin Tarantino", "genre": "crime", "rating": 8.9},
    },
    {
        "page_content": "The animated movie 'Spirited Away' by Hayao Miyazaki was released in 2001 and is a fantasy film with a rating = 8.6. It's about a young girl venturing into the spirit world.",
        "metadata": {"year": 2001, "director": "Hayao Miyazaki", "genre": "fantasy", "rating": 8.6},
    },
    {
        "page_content": "The science fiction movie 'Dune: Part Two' was released in 2024 by Denis Villeneuve with a rating = 8.8. It continues the story of Paul Atreides on Arrakis.",
        "metadata": {"year": 2024, "director": "Denis Villeneuve", "genre": "science fiction", "rating": 8.8},
    },
    {
        "page_content": "The comedy 'The Big Lebowski' from 1998 directed by Coen Brothers has a rating = 7.8. It's a cult classic about a slacker embroiled in a kidnapping plot.",
        "metadata": {"year": 1998, "director": "Coen Brothers", "genre": "comedy", "rating": 7.8},
    },
]

# Convert dictionaries to Document objects
docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in raw_docs]

# Initialize ChromaDB to persist data locally
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="movies_collection",
    persist_directory="./chroma_db"
)

# --- 4. Define Document Content and Metadata Schema ---
# Refined instruction to guide the LLM on the exact filter format
document_content_description = (
    "A movie synopsis and its properties. Queries should identify the movie genre, year, director, or rating. "
    "The 'filter' field in the output JSON should be a **string** representing a logical expression, "
    "using functions like 'eq', 'gt', 'lt', 'gte', 'lte', 'and', 'or' on attributes: 'genre', 'year', 'director', 'rating'. "
    "Example filter string: 'and(eq(\"genre\", \"science fiction\"), gt(\"year\", 2010))'"
)
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie, e.g., science fiction, thriller, crime, fantasy, comedy",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The release year of the movie",
        type="integer",
    ),
    AttributeInfo(
        name="director",
        description="The director of the movie",
        type="string",
    ),
    AttributeInfo(
        name="rating",
        description="The IMDb rating of the movie (out of 10)",
        type="float",
    ),
]

# --- 5. Initialize the LLM (for a truly offline system, use a local LLM like Ollama) ---
# To use Ollama:
# 1. Download and install Ollama from https://ollama.com/
# 2. Pull a model, e.g., `ollama pull llama2` or `ollama pull phi3`
# 3. Ensure Ollama is running in the background.
llm = ChatOllama(model="llama2") # Replace "llama2" with your desired local model name
# If you want to switch back to Groq (online), uncomment the line below and re-add Groq API key:
# llm = ChatGroq(temperature=0, model_name="meta-llama/llama-4-scout-17b-16e-instruct")

# --- 6. Create the SelfQueryRetriever ---
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# --- 7. Perform Self-Queries ---
print("\n--- Query 1: Science fiction movies released after 2010 ---")
query_1 = "What science fiction movies were released after 2010?"
docs_1 = retriever.invoke(query_1)
for doc in docs_1:
    print(f"Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")

print("\n--- Query 2: Thriller movies by Christopher Nolan ---")
query_2 = "Are there any thriller movies by Christopher Nolan?"
docs_2 = retriever.invoke(query_2)
for doc in docs_2:
    print(f"Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")

print("\n--- Query 3: Movies with a rating above 8.5 ---")
query_3 = "Which movies have a rating above 8.5?"
docs_3 = retriever.invoke(query_3)
for doc in docs_3:
    print(f"Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")

print("\n--- Query 4: Comedy movies ---")
query_4 = "Tell me about comedy movies."
docs_4 = retriever.invoke(query_4)
for doc in docs_4:
    print(f"Content: {doc.page_content}")
    print(f"  Metadata: {doc.metadata}\n")
