import pymongo
from typing import Dict, List
from config import MONGO_URI, PDF_DB_NAME, PDF_COLLECTION_NAME

def get_merged_pdf_chunks() -> Dict[str, str]:
    """
    Connects to MongoDB, retrieves PDF chunks, and merges them with
    2 preceding and 2 succeeding chunks for context.
    Returns a dictionary where keys are original chunk _ids (as strings)
    and values are the merged text.
    """
    mongo_client_pdf = None
    merged_texts = {}
    try:
        mongo_client_pdf = pymongo.MongoClient(MONGO_URI)
        pdf_db = mongo_client_pdf[PDF_DB_NAME]
        pdf_collection = pdf_db[PDF_COLLECTION_NAME]

        pages_cursor = pdf_collection.distinct("page_number")
        pages = sorted(list(pages_cursor))

        for page_num in pages:
            chunks_on_page = list(pdf_collection.find(
                {"page_number": page_num},
                {"_id": 1, "chunk_number": 1, "chunk_text": 1}
            ).sort("chunk_number", pymongo.ASCENDING))

            for i in range(len(chunks_on_page)):
                current_doc = chunks_on_page[i]
                current_id = str(current_doc["_id"])

                surrounding_chunks_docs = chunks_on_page[max(i - 2, 0):min(i + 3, len(chunks_on_page))]
                merged_text = " ".join([doc["chunk_text"] for doc in surrounding_chunks_docs])
                merged_texts[current_id] = merged_text
    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error for PDF Chunks: {e}")
        print("Please ensure your MongoDB server is running on localhost:27017.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during PDF chunks processing: {e}")
    finally:
        if mongo_client_pdf:
            mongo_client_pdf.close()
            print("MongoDB PDF chunks connection closed.")
    return merged_texts

def get_first_pdf_document():
    """
    Retrieves the first document from the pdf_chunks collection based on
    page_number and chunk_number.
    """
    mongo_client_pdf = None
    first_pdf_doc = None
    try:
        mongo_client_pdf = pymongo.MongoClient(MONGO_URI)
        pdf_db = mongo_client_pdf[PDF_DB_NAME]
        pdf_collection = pdf_db[PDF_COLLECTION_NAME]
        first_pdf_doc = pdf_collection.find_one({}, sort=[('page_number', pymongo.ASCENDING), ('chunk_number', pymongo.ASCENDING)])
    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error when fetching first PDF document: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred when fetching first PDF document: {e}")
    finally:
        if mongo_client_pdf:
            mongo_client_pdf.close()
    return first_pdf_doc

def get_all_pdf_documents_details() -> List[Dict]:
    """
    Retrieves all documents from the pdf_chunks collection, sorted by page_number and chunk_number.
    Returns a list of dictionaries, where each dictionary is a document.
    """
    mongo_client_pdf = None
    all_pdf_docs = []
    try:
        mongo_client_pdf = pymongo.MongoClient(MONGO_URI)
        pdf_db = mongo_client_pdf[PDF_DB_NAME]
        pdf_collection = pdf_db[PDF_COLLECTION_NAME]
        all_pdf_docs = list(pdf_collection.find({}, sort=[('page_number', pymongo.ASCENDING), ('chunk_number', pymongo.ASCENDING)]))
    except pymongo.errors.ConnectionFailure as e:
        print(f"❌ MongoDB connection error when fetching all PDF documents: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred when fetching all PDF documents: {e}")
    finally:
        if mongo_client_pdf:
            mongo_client_pdf.close()
            print("MongoDB PDF documents details connection closed.")
    return all_pdf_docs