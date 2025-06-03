from langgraph.graph import START, END, StateGraph
from models import State
from llm_init import llm, eval_llm
from knowledge_base import knowledge_list, retriever
from agents import load_agents_from_mongo, available_agents
from workflow_nodes import main_node, final_report_generator
from pdf_processor import get_merged_pdf_chunks, get_first_pdf_document, get_all_pdf_documents_details # MODIFIED: Import new function
from config import AGENTS_DB_NAME, AGENTS_COLLECTION_NAME
from database_saver import save_results_to_mongo, clear_results_collection

def run_workflow():
    # Clear the results collection at the beginning of each program execution
    clear_results_collection()

    # Load agents dynamically from MongoDB
    print("Loading agents from MongoDB...")
    load_agents_from_mongo(llm, eval_llm)

    # Initialize the StateGraph with the defined State
    graph_builder = StateGraph(State)

    # Add the main_node and final_report_generator nodes to the graph
    graph_builder.add_node("main_node", main_node)
    graph_builder.add_node("fnl_rprt", final_report_generator)

    # Set the entry point of the graph to "main_node"
    graph_builder.add_edge(START, "main_node")

    # Dynamically add edges from "main_node" to each loaded agent, and then from each agent to the "fnl_rprt"
    for agent_name in available_agents:
        graph_builder.add_node(agent_name, available_agents[agent_name])
        graph_builder.add_edge("main_node", agent_name)
        graph_builder.add_edge(agent_name, "fnl_rprt")

    # Set the exit point of the graph to "fnl_rprt"
    graph_builder.add_edge("fnl_rprt", END) 

    # Compile the graph for execution
    graph = graph_builder.compile()

    # --- PDF CHUNKS PROCESSING ---
    merged_texts = get_merged_pdf_chunks() # This function already gets all merged chunks

    # --- CHOOSE ONE OPTION BELOW: Process first chunk OR Process all chunks ---

    # OPTION 1: Process only the first PDF chunk (COMMENTED OUT)
    print("\n--- OPTION 1: Executing graph.invoke() for the first database entry ---")
    first_pdf_doc = get_first_pdf_document() # Gets details of the first document
    documents_to_process = [first_pdf_doc] if first_pdf_doc else []


    # OPTION 2: Process all PDF chunks (ACTIVE BY DEFAULT)
    #print("\n--- OPTION 2: Executing graph.invoke() for ALL database entries ---")
    #documents_to_process = get_all_pdf_documents_details() # Get details for all documents


    # --- Common processing loop for selected documents ---
    if documents_to_process:
        for doc_to_process in documents_to_process:
            if not doc_to_process: # Skip if doc_to_process is None
                continue

            row_id_mongo = str(doc_to_process.get("_id"))
            book_title = doc_to_process.get("book_title")
            page_number = doc_to_process.get("page_number")
            chunk_number = doc_to_process.get("chunk_number")
            original_chunk_text = doc_to_process.get("chunk_text")

            merged_text_for_id = merged_texts.get(row_id_mongo, original_chunk_text) # Get merged text using chunk's ID

            print(f"\n--- Processing ID: {row_id_mongo} (Page: {page_number}, Paragraph: {chunk_number}) ---")
            print(f"Original Chunk Text: {original_chunk_text}")
            print(f"Merged Context Text: {merged_text_for_id}\n")

            report_data = {
                "report_text": merged_text_for_id,
                "metadata": {
                    "page": page_number,
                    "paragraph": chunk_number,
                    "title": book_title,
                    "chunk_id": row_id_mongo # IMPORTANT: Pass the MongoDB _id as chunk_id for saving
                }
            }

            print(f"\n--- Langgraph Workflow Input for ID: {row_id_mongo} ---")
            print("Initial state before agent execution. Individual agents will now perform their internal evaluation loops.")
            print("-" * 40)

            result_with_review = graph.invoke(report_data)

            # Save the results to MongoDB after the workflow completes for this chunk
            save_results_to_mongo(report_data, result_with_review)

            print("\n--- Langgraph Workflow Final Output ---")
            for agent_name, agent_output_data in result_with_review["main_node_output"].items():
                print(f"\n--- Summary for {agent_name} ---")
                print(f"  Parsed Output: {agent_output_data['output']}")
                print(f"  Confidence: {agent_output_data['confidence']}%")
                print(f"  Retries: {agent_output_data['retries']}")
                print(f"  Human Review Needed: {agent_output_data['human_review']}")
            print("\nFull Result Dictionary (for debugging):")
            print(result_with_review)
            print("-" * 40)

    else:
        print("No documents found to process. Please ensure your PDF data is in MongoDB.")

if __name__ == "__main__":
    run_workflow()