import os
from dotenv import load_dotenv
load_dotenv()
# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
AGENTS_DB_NAME = os.getenv("MONGO_DB_Agent")
AGENTS_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

KB_DB_NAME =os.getenv("MONGO_DB_KB")
KB_COLLECTION_NAME =os.getenv("MONGO_COLLECTION_KB")

PDF_DB_NAME = os.getenv("MONGO_DB_PDF")
PDF_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_PDF")

# --- ChromaDB Persistent Directory ---
CHROMA_DB_DIRECTORY = "chrome_dB"

# Replace with your actual API keys
GROQ_API_KEY=os.getenv("GROQ_API_KEY1")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY1")


