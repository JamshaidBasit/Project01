from typing import Dict
from models import State
from agents import available_agents

# ─── CORE WORKFLOW NODES ─────────────────────────────────────────────────────
def main_node(state: State) -> Dict:
    """
    The initial node in the graph. It typically receives the initial state
    and can perform any setup or initial processing before routing to other agents.
    """
    print("main_node called")
    return {}

def final_report_generator(state: State) -> Dict:
    """
    Aggregates the outputs from all review agents and generates a comprehensive
    final decision report.
    """
    print("\n--- Final Report Generator Called ---")
    report_parts = {}

    for agent_name in available_agents:
        result = state["main_node_output"].get(agent_name, {})
        report_parts[agent_name] = result

    aggregate_history = state.get("aggregate", [])
    final_decision_report = "*Review Report:*\n\n"

    for name, result in report_parts.items():
        output_content = result.get("output", {}).get("problematic_text", "N/A")
        if "error" in result.get("output", {}):
            output_content = f"ERROR: {result['output']['error']}"

        confidence = result.get("confidence", 0)
        retries = result.get("retries", 0)
        human_review = result.get("human_review", False)

        final_decision_report += (
            f"*{name} Review:*\n"
            f"  Output: {output_content}\n"
            f"  Confidence: {confidence}%\n"
            f"  Retries: {retries}\n"
            f"  Human Review Needed: {human_review}\n\n"
        )

    final_decision_report += (
        "*Aggregate History:*\n" +
        "\n".join(aggregate_history) +
        "\n\nThis report compiles insights from various critical reviews."
    )
    print("Final Decision Report Generated.")
    print("-" * 30)
    return {"final_decision_report": final_decision_report}