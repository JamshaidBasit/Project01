from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from config import GROQ_API_KEY, GOOGLE_API_KEY
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize Google Generative AI Embeddings for vector store creation
embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("embeddings_model"), google_api_key=GOOGLE_API_KEY)

# ─── LLM INITIALIZATION ──────────────────────────────────────────────────────
# Initialize the ChatGroq language model for review agents
llm = ChatGroq(temperature=0, model_name=os.getenv("llm_groq"), groq_api_key=GROQ_API_KEY)
# Use a separate LLM for evaluation if desired, or reuse the main LLM
eval_llm = ChatGroq(temperature=0.2, model_name=os.getenv("eval_llm1"), groq_api_key=GROQ_API_KEY)

# Initialize a separate Google LLM for self-querying to avoid conflicts with Groq LLM
llm_google = ChatGoogleGenerativeAI(model=os.getenv("llm_google1"), google_api_key=GOOGLE_API_KEY)