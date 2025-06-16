

import pymongo
from typing import Dict
from config import MONGO_URI
import os
from dotenv import load_dotenv
load_dotenv()
# --- MongoDB Configuration for Final Results ---
RESULTS_DB_NAME = os.getenv("RESULTS_DB_NAME1")
RESULTS_COLLECTION_NAME = os.getenv("RESULTS_COLLECTION_NAM1")


def clear_results_collection():
    """
    Clears all documents from the RESULTS_COLLECTION_NAME in RESULTS_DB_NAME.
    This should be called at the beginning of a program execution to ensure a fresh start.
    """
    mongo_client = None
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        results_db = mongo_client[RESULTS_DB_NAME]
        results_collection = results_db[RESULTS_COLLECTION_NAME]

        # Delete all documents in the collection
        delete_result = results_collection.delete_many({})
        print(f"🧹 Cleared {delete_result.deleted_count} documents from '{RESULTS_DB_NAME}.{RESULTS_COLLECTION_NAME}'.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error during clearing: {e}")
        print("Please ensure your MongoDB server is running on localhost:27017.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while clearing results collection: {e}")
    finally:
        if mongo_client:
            mongo_client.close()

def save_results_to_mongo(report_data: Dict, result_with_review: Dict):
    """
    Saves the final analysis results for a single text chunk to MongoDB.

    Args:
        report_data (Dict): Contains information about the text chunk (e.g., ID, page, text, metadata, predicted_label).
        result_with_review (Dict): The complete output from the Langgraph workflow,
                                   including each agent's detailed review.
    """
    mongo_client = None
    try:
        mongo_client = pymongo.MongoClient(MONGO_URI)
        results_db = mongo_client[RESULTS_DB_NAME]
        results_collection = results_db[RESULTS_COLLECTION_NAME]

        # Extract base information from report_data
        document_id = report_data.get("metadata", {}).get("chunk_id", "N/A")
        book_title = report_data.get("metadata", {}).get("title", "N/A")
        page_number = report_data.get("metadata", {}).get("page", "N/A")
        chunk_number = report_data.get("metadata", {}).get("paragraph", "N/A")
        analyzed_text = report_data.get("report_text", "N/A")
        predicted_label = report_data.get("metadata", {}).get("predicted_label", "N/A") # NEW
        classification_scores = report_data.get("metadata", {}).get("classification_scores", {}) # NEW

        # Prepare the base document to be saved
        result_document = {
            "ID": document_id,
            "Book Name": book_title,
            "Page Number": page_number,
            "Chunk no.": chunk_number,
            "Text Analyzed": analyzed_text,
            "Predicted Label": predicted_label, # NEW
            "Classification Scores": classification_scores, # NEW
        }

        # Add agent-specific results dynamically
        main_node_output = result_with_review.get("main_node_output", {})
        for agent_name, agent_data in main_node_output.items():
            agent_output = agent_data.get("output", {})
            
            # Extract specific fields from the agent's output
            agent_result = {
                "issue_found": agent_output.get("issues_found", False),
                "problematic_text": agent_output.get("problematic_text", ""),
                "observation": agent_output.get("observation", ""),
                "recommendation": agent_output.get("recommendation", ""),
                "confidence": agent_data.get("confidence", 0),
                "human_review": agent_data.get("human_review", False),
                "retries": agent_data.get("retries", 0)
            }
            result_document[agent_name] = agent_result

        # Insert the document into MongoDB
        results_collection.insert_one(result_document)
        print(f"✅ Analysis results for chunk ID '{document_id}' saved to MongoDB in '{RESULTS_DB_NAME}.{RESULTS_COLLECTION_NAME}'.")

    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error: {e}")
        print("Please ensure your MongoDB server is running on localhost:27017.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while saving results to MongoDB: {e}")
    finally:
        if mongo_client:
            mongo_client.close()