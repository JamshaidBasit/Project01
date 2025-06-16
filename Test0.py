import os
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import numpy as np  # Import NumPy

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.embeddings import Embeddings # <--- ADD THIS LINE
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_groq import ChatGroq # Import ChatGroq

# --- 1. Custom Embedding Class for Qwen3-Embedding-0.6B (FIXED) ---
class Qwen3Embeddings(Embeddings):
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model_name).to(self.device)
        # Tokenizer is usually handled internally by SentenceTransformer,
        # but we keep it if you need direct access for other tasks.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # model.encode by default returns numpy arrays, or torch tensors if convert_to_numpy=False.
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False) # Ensure numpy array

        # Ensure it's a numpy array, then convert to list.  This handles both cases.
        return np.array(embeddings).tolist()

    def embed_query(self, text: str) -> List[float]:
        # For queries, use the "query" prompt for optimal performance with Qwen3
        embedding = self.model.encode([text], prompt_name="query", convert_to_numpy=True) # Ensure numpy array

        # embedding will be a numpy array of shape (1, embedding_dim)
        # We need to get the single embedding vector and then convert it to a list
        return np.array(embedding[0]).tolist()


# --- Configuration for Groq ---
os.environ["GROQ_API_KEY"] = "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" # <--- Replace with your actual Groq API key

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# Initialize Qwen3 Embeddings
qwen_embeddings = Qwen3Embeddings()

# --- 2. Prepare Documents with Metadata (Remains the same) ---
docs = [
    Document(
        page_content="The latest iPhone, the iPhone 15 Pro Max, features a powerful A17 Bionic chip and a titanium design. It was released in September 2023.",
        metadata={"source": "Apple website", "year": 2023, "product": "iPhone", "type": "smartphone"}
    ),
    Document(
        page_content="Samsung's flagship foldable phone, the Galaxy Z Fold 5, offers a large inner display and enhanced multitasking capabilities. It was released in August 2023.",
        metadata={"source": "Samsung newsroom", "year": 2023, "product": "Galaxy Z Fold", "type": "smartphone"}
    ),
    Document(
        page_content="Google's Pixel 8 Pro comes with an excellent camera system powered by Google's Tensor G3 chip. Released in October 2023.",
        metadata={"source": "Google blog", "year": 2023, "product": "Pixel", "type": "smartphone"}
    ),
    Document(
        page_content="The MacBook Air M3, released in March 2024, is known for its incredible battery life and fanless design.",
        metadata={"source": "Apple press release", "year": 2024, "product": "MacBook Air", "type": "laptop"}
    ),
    Document(
        page_content="Microsoft Surface Laptop Studio 2, launched in September 2023, is a versatile device for creatives and professionals.",
        metadata={"source": "Microsoft news", "year": 2023, "product": "Surface Laptop Studio", "type": "laptop"}
    ),
    Document(
        page_content="The newest PlayStation 5 Slim offers a more compact design and includes a detachable disc drive. It was released in November 2023.",
        metadata={"source": "PlayStation blog", "year": 2023, "product": "PlayStation 5", "type": "gaming console"}
    ),
]

# --- 3. Choose a Vector Store and Add Documents (Remains the same) ---
vectorstore = Chroma.from_documents(docs, qwen_embeddings)

# --- 4. Define Metadata Field Information (Remains the same) ---
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The source of the information (e.g., 'Apple website', 'Samsung newsroom')",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="The year the product was released",
        type="integer",
    ),
    AttributeInfo(
        name="product",
        description="The name of the product (e.g., 'iPhone', 'MacBook Air')",
        type="string",
    ),
    AttributeInfo(
        name="type",
        description="The category of the product (e.g., 'smartphone', 'laptop', 'gaming console')",
        type="string",
    ),
]

document_content_description = "Brief summary of a tech product"

# --- 5. Instantiate the Self-Query Retriever (Using Groq LLM) ---
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True,
    enable_limit=True
)

# --- Example Queries (English and Urdu) ---
queries = [
    "What are some smartphones released in 2023?",
    "Tell me about laptops released in 2024.",
    "Which gaming consoles were released in 2023?",
    "Find me any product from Apple.",
    "What are some products released in 2023 that are not smartphones?",
    "Give me the top 1 laptop released in 2023.",
    "2023 میں ریلیز ہونے والے کچھ اسمارٹ فونز کون سے ہیں؟", # What are some smartphones released in 2023?
    "2024 میں ریلیز ہونے والے لیپ ٹاپس کے بارے میں بتائیں؟", # Tell me about laptops released in 2024?
    "2023 میں کون سے گیمنگ کنسولز ریلیز ہوئے تھے؟", # Which gaming consoles were released in 2023?
    "ایپل کی کوئی بھی پروڈکٹ تلاش کریں؟", # Find me any product from Apple.
    "2023 میں ریلیز ہونے والے ایسے پروڈکٹس کون سے ہیں جو اسمارٹ فونز نہیں ہیں؟", # What are some products released in 2023 that are not smartphones?
    "2023 میں ریلیز ہونے والے سب سے بہترین 1 لیپ ٹاپ کا نام بتائیں؟", # Give me the top 1 laptop released in 2023.
]

print("--- Running Self-Queries with Groq ---")
for i, query in enumerate(queries):
    print(f"\nQuery {i+1}: {query}")
    try:
        results = retriever.invoke(query)
        for j, doc in enumerate(results):
            print(f"  Result {j+1}: {doc.page_content} (Source: {doc.metadata.get('source')}, Year: {doc.metadata.get('year')})")
    except Exception as e:
        print(f"  Error processing query: {e}")