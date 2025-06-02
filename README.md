# Multi-Agent PDF Document Analysis Workflow
## Project Structure
.
├── config.py             # Global configurations (API keys, MongoDB URIs, DB/Collection names)
├── models.py             # Defines the LangGraph State (TypedDict) for the workflow
├── llm_init.py           # Initializes and configures LLM instances (Groq, Gemini)
├── knowledge_base.py     # Manages external knowledge (ChromaDB, retriever setup)
├── agents.py             # Defines the structure and behavior of individual AI agents; handles dynamic loading
├── workflow_nodes.py     # Contains the LangGraph nodes (main processing logic, final report generation)
├── pdf_processor.py      # Handles PDF extraction, chunking, and contextual merging
├── database_saver.py     # Manages saving analysis results to MongoDB and clearing the collection
├── main.py               # The main script to run the multi-agent workflow
└── requirements.txt      # Lists all project dependencie
# Setup
## Prerequisites
Python 3.9+

MongoDB: Ensure MongoDB Community Server is installed and running on localhost:27017.

Download MongoDB

MongoDB Installation Guide

## Installation
Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

## Create a virtual environment (recommended):
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

## Install dependencies:
pip install -r requirements.txt

## API Keys
You will need API keys for the LLMs used in this project.

## Groq API Key:

Sign up at Groq Console.

Generate an API key.

Open config.py and replace "your_groq_api_key_here" with your actual Groq API key:
GROQ_API_KEY = "your_groq_api_key_here"

## Google Gemini API Key:

Get your API key from Google AI Studio.

Open config.py and replace "your_gemini_api_key_here" with your actual Google Gemini API key:
GOOGLE_API_KEY = "your_gemini_api_key_here"


# Usage
## Running the Workflow
To store document chunks run 
python MongoDatabase_for_pdf.py

Once all prerequisites are met and configurations are set up, run the main.py script:

python main.py
