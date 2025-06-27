from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from config import GROQ_API_KEY
import os
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
load_dotenv()

# --- 2. Define your Embedding Model (BGE-Large with FastEmbed) ---
embeddings = FastEmbedEmbeddings(model_name=os.getenv("embeddings_model"))

# ─── LLM INITIALIZATION ──────────────────────────────────────────────────────
# Initialize the ChatGroq language model for review agents
llm = ChatGroq(temperature=0, model_name=os.getenv("llm_groq"), groq_api_key=GROQ_API_KEY)
# Use a separate LLM for evaluation if desired, or reuse the main LLM
eval_llm = ChatGroq(temperature=0.2, model_name=os.getenv("eval_llm1"), groq_api_key=GROQ_API_KEY)
## For Embedding
llm1 = ChatGroq(
    temperature=0,
    model_name=os.getenv("eval_llm1"),groq_api_key=GROQ_API_KEY
)

