import json
import pymongo
from typing import List, Dict, Callable
from langgraph.graph import END, StateGraph
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from models import State
from knowledge_base import get_relevant_info, retriever, knowledge_list
from config import MONGO_URI, AGENTS_DB_NAME, AGENTS_COLLECTION_NAME

# Define a type for agent functions for clear type hinting
Agent = Callable[[State], Dict]

# Define a dictionary to hold all available agents, mapping names to their functions
available_agents: Dict[str, Agent] = {}

# ─── PROMPT TEMPLATE ────────────────────────────────────────────────────────
TEMPLATE = """
You are an expert alignment reviewer for books about Pakistan's history and politics.
Your task is to identify content that may not align with Pakistan's official policies, narratives, or national interests.

Current text to analyze:

{text}

Page: {page}
Paragraph: {paragraph}

Context: This text is from a book titled "{title}".
Predicted Category: {predicted_label}

Your task is to determine if this text contains any content that:
{specific_criteria}

If you find any issues, provide:
1. The specific problematic text
2. A brief observation explaining why it's problematic
3. A recommendation (delete, rephrase, fact-check, or provide references)

Follow these guidelines:
- Be objective in your assessment
- Consider the national interest of Pakistan
- Focus only on real alignment issues, not stylistic concerns
- Use the official policy guidelines and Knowledge Base to inform your judgment

Knowledge Base:
Official Narrative: {official_narrative}
Key Points: {key_points}
Sensitive Aspects: {sensitive_aspects}
Recommended Terminology: {recommended_terminology}
Authoritative Sources: {authoritative_sources}

Official policy guidelines:
{policy_guidelines}

Respond in JSON format and must follow JSON Formate:
{{
    "issues_found": true/false,
    "problematic_text": "exact text that is problematic",
    "observation": "brief explanation of the issue",
    "recommendation": "delete/rephrase/fact-check/provide references"
}}

Respond only with valid JSON.Do not include any explanation outsides the JSON block.
"""

def register_agent(name: str, agent_function: Agent):
    """
    Registers an agent function under a given name in the global available_agents dictionary.
    """
    available_agents[name] = agent_function

def create_review_agent(review_name: str, criteria: str, guidelines: str, llm_model: ChatGroq, eval_llm_model: ChatGroq) -> Agent:
    """
    Creates a specialized review agent function that includes an internal evaluation loop.
    """
    prompt_template = PromptTemplate.from_template(TEMPLATE)

    def agent_sub_step(state: State) -> State:
        print(f"\n--- {state['current_agent_name']} Sub-Agent Step - Attempt {state.get('current_agent_retries', 0) + 1} ---")
        report_text = state["report_text"]
        metadata = state["metadata"]
        predicted_label = metadata.get("predicted_label", "N/A")

        official_narrative = "No relevant official narrative found."
        key_points_str = "No relevant key points found."
        sensitive_aspects_str = "No relevant sensitive aspects found."
        recommended_terminology_str = "No relevant recommended terminology found."
        authoritative_sources_str = "No relevant authoritative sources found."

        # MODIFICATION START: Conditional KB application based on agent name and predicted label
        # FactCheckingReview agent always uses KB
        if state['current_agent_name'] == "FactCheckingReview":
            print(f"--- {state['current_agent_name']} (FactCheckingReview) always uses KB. ---")
            relevant_knowledge = get_relevant_info(report_text)
            if isinstance(relevant_knowledge, list) and relevant_knowledge:
                relevant_item = relevant_knowledge[0]
                official_narrative = relevant_item.get("official_narrative", official_narrative)
                key_points_str = ", ".join(relevant_item.get("key_points", []))
                sensitive_aspects_str = json.dumps(relevant_item.get("sensitive_aspects", []))
                recommended_terminology_str = json.dumps(relevant_item.get("recommended_terminology", {}))
                authoritative_sources_str = ", ".join(relevant_item.get("authoritative_sources", []))
        # Other agents skip KB if the text is classified as "general or unrelated text"
        elif predicted_label == "general or unrelated text":
            print(f"--- Skipping KB lookup for '{predicted_label}' chunk for {state['current_agent_name']}. ---")
            official_narrative = "N/A (Text classified as general or unrelated)"
            key_points_str = "N/A (Text classified as general or unrelated)"
            sensitive_aspects_str = "N/A (Text classified as general or unrelated)"
            recommended_terminology_str = "N/A (Text classified as general or unrelated)"
            authoritative_sources_str = "N/A (Text classified as general or unrelated)"
        # All other cases (not FactCheckingReview and not "general or unrelated text") use KB
        else:
            print(f"--- Using KB for '{predicted_label}' chunk for {state['current_agent_name']}. ---")
            relevant_knowledge = get_relevant_info(report_text)
            if isinstance(relevant_knowledge, list) and relevant_knowledge:
                relevant_item = relevant_knowledge[0]
                official_narrative = relevant_item.get("official_narrative", official_narrative)
                key_points_str = ", ".join(relevant_item.get("key_points", []))
                sensitive_aspects_str = json.dumps(relevant_item.get("sensitive_aspects", []))
                recommended_terminology_str = json.dumps(relevant_item.get("recommended_terminology", {}))
                authoritative_sources_str = ", ".join(relevant_item.get("authoritative_sources", []))
        # MODIFICATION END

        prompt = prompt_template.format(
            text=report_text,
            page=metadata.get("page", "N/A"),
            paragraph=metadata.get("paragraph", "N/A"),
            title=metadata.get("title", "N/A"),
            predicted_label=predicted_label,
            specific_criteria=criteria,
            policy_guidelines=guidelines,
            official_narrative=official_narrative,
            key_points=key_points_str,
            sensitive_aspects=sensitive_aspects_str,
            recommended_terminology=recommended_terminology_str,
            authoritative_sources=authoritative_sources_str
        )

        print(f"--- {state['current_agent_name']} Input Prompt ---")
        print(prompt)
        print("-" * 30)

        response = llm_model.invoke(prompt)
        raw_output = response.content
        parsed_output = {}
        human_review_flag = False

        try:
            parsed_output = json.loads(raw_output)
            print(f"--- {state['current_agent_name']} Raw Output ---")
            print(raw_output)
            print(f"--- {state['current_agent_name']} Parsed Output ---")
            print(parsed_output)
            print("-" * 30)
        except json.JSONDecodeError:
            error_message = f"Error decoding JSON from {state['current_agent_name']}: {raw_output}"
            print(f"--- {state['current_agent_name']} Output (Error) ---")
            print(error_message)
            print("-" * 30)
            parsed_output = {"error": error_message, "human_review_reason": "JSON_DECODE_ERROR"}
            human_review_flag = True

        return {
            "current_agent_input_prompt": prompt,
            "current_agent_raw_output": raw_output,
            "current_agent_parsed_output": parsed_output,
            "current_agent_retries": state.get("current_agent_retries", 0) + 1,
            "current_agent_human_review": human_review_flag
        }

    def evaluation_sub_step(state: State) -> State:
        if state.get("current_agent_human_review", False):
            print(f"\n--- {state['current_agent_name']} Evaluation Skipped: JSON Decode Error ---")
            return {"current_agent_confidence": 0}

        print(f"\n--- {state['current_agent_name']} Sub-Evaluation Step - Attempt {state.get('current_agent_retries', 0)} ---")
        prompt_from_agent = state["current_agent_input_prompt"]
        response_from_agent = state["current_agent_raw_output"]

        eval_prompt = f"""
Evaluate the following:

Prompt given to agent:
{prompt_from_agent}

Agent's Raw Response:
"{response_from_agent}"

How correct and relevant is the Response to the Prompt?

Give a confidence score between 0 and 100.

Respond only with a single, valid JSON object that follows this structure:
{{"confidence": <score>}}
DO NOT include any explanation or text outside of the JSON object.
"""
        print(f"--- {state['current_agent_name']} Evaluation Prompt ---")
        print(eval_prompt)
        print("-" * 30)

        eval_response = eval_llm_model.invoke(eval_prompt).content
        confidence = 0
        try:
            eval_data = json.loads(eval_response)
            confidence = int(eval_data.get("confidence", 0))
            print(f"--- {state['current_agent_name']} Evaluation Response ---")
            print(eval_data)
            print(f"Confidence: {confidence}")
            print("-" * 30)
        except json.JSONDecodeError:
            print(f"--- {state['current_agent_name']} Evaluation Response (Error) ---")
            print(f"Error decoding JSON from evaluator: {eval_response}")
            print("Confidence set to 0 due to evaluation parsing error.")
            print("-" * 30)
            confidence = 0
        except Exception as e:
            print(f"--- {state['current_agent_name']} Evaluation Response (Other Error) ---")
            print(f"An unexpected error occurred during evaluation parsing: {e}")
            print("Confidence set to 0.")
            print("-" * 30)
            confidence = 0

        return {
            "current_agent_confidence": confidence
        }

    def route_sub_step(state: State) -> str:
        if state.get("current_agent_human_review", False):
            print(f"\n⚠️ {state['current_agent_name']} JSON Decode Error detected. Routing directly to human review.")
            return "human_review_needed_sub_step"

        max_retries = 3
        if state["current_agent_confidence"] >= 80:
            print(f"\n✨ {state['current_agent_name']} Confidence {state['current_agent_confidence']}% is sufficient. Exiting sub-workflow.")
            return "end"
        elif state["current_agent_retries"] < max_retries:
            print(f"\n🔄 {state['current_agent_name']} Confidence {state['current_agent_confidence']}% is too low. Retrying... (Attempt {state['current_agent_retries']} of {max_retries})")
            return "agent_sub_step"
        else:
            print(f"\n⚠️ {state['current_agent_name']} Confidence {state['current_agent_confidence']}% still too low after {max_retries} retries. Human review needed for this agent's output.")
            return "human_review_needed_sub_step"

    def human_review_sub_step(state: State) -> State:
        print(f"\n--- {state['current_agent_name']} Human Review Needed ---")
        return {
            "current_agent_human_review": True
        }

    agent_graph_builder = StateGraph(State)
    agent_graph_builder.add_node("agent_sub_step", RunnableLambda(agent_sub_step))
    agent_graph_builder.add_node("evaluation_sub_step", RunnableLambda(evaluation_sub_step))
    agent_graph_builder.add_node("human_review_needed_sub_step", RunnableLambda(human_review_sub_step))

    agent_graph_builder.set_entry_point("agent_sub_step")
    agent_graph_builder.add_edge("agent_sub_step", "evaluation_sub_step")
    agent_graph_builder.add_conditional_edges(
        "evaluation_sub_step",
        route_sub_step,
        {
            "agent_sub_step": "agent_sub_step",
            "human_review_needed_sub_step": "human_review_needed_sub_step",
            "end": END
        }
    )
    agent_graph_builder.add_edge("human_review_needed_sub_step", END)

    agent_sub_graph = agent_graph_builder.compile()

    def review_agent_with_evaluation(state: State) -> Dict:
        initial_sub_state = {
            "report_text": state["report_text"],
            "metadata": state["metadata"],
            "current_agent_name": review_name,
            "current_agent_retries": 0,
            "current_agent_confidence": 0,
            "current_agent_human_review": False,
            "final_decision_report": "",
            "aggregate": [],
            "main_node_output": {}
        }

        final_sub_state = agent_sub_graph.invoke(initial_sub_state)

        agent_result = final_sub_state.get("current_agent_parsed_output", {"error": "No output parsed"})
        agent_confidence = final_sub_state.get("current_agent_confidence", 0)
        agent_retries = final_sub_state.get("current_agent_retries", 0)
        agent_human_review = final_sub_state.get("current_agent_human_review", False)

        return {
            review_name: agent_result,
            "aggregate": [f"{review_name} Output: {agent_result} (Confidence: {agent_confidence}%, Retries: {agent_retries}, Human Review: {agent_human_review})"],
            "main_node_output": {
                review_name: {
                    "output": agent_result,
                    "confidence": agent_confidence,
                    "retries": agent_retries,
                    "human_review": agent_human_review
                }
            }
        }
    return review_agent_with_evaluation

def load_agents_from_mongo(llm_model: ChatGroq, eval_llm_model: ChatGroq):
    """
    Loads agent configurations (name, criteria, guidelines) from a MongoDB collection
    and registers them as review agents.
    """
    client = None
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[AGENTS_DB_NAME]
        collection = db[AGENTS_COLLECTION_NAME]

        rows = collection.find({})

        for doc in rows:
            agent_name = doc.get("agent_name")
            criteria = doc.get("criteria")
            guidelines = doc.get("guidelines")

            if agent_name and criteria and guidelines:
                agent = create_review_agent(agent_name, criteria, guidelines, llm_model, eval_llm_model)
                register_agent(agent_name, agent)
                print(f"Agent '{agent_name}' loaded from MongoDB.")
            else:
                print(f"Error: Missing 'criteria' or 'guidelines' for agent '{agent_name}' in MongoDB document: {doc}")
    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB connection error for Agents: {e}")
        print("Please ensure MongoDB server is running on localhost:27017.")
    except Exception as e:
        print(f"An unexpected error occurred during agent loading: {e}")
    finally:
        if client:
            client.close()