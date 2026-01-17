from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    load_dataset,
    profile_dataset,
    understand_goal,
    validate_alignment,
    compute_score,
    decide_outcome,
    generate_agent_answer,
    build_response
)

def create_validation_graph():
    """Create LangGraph workflow for validation"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("load_dataset", load_dataset)
    workflow.add_node("profile_dataset", profile_dataset)
    workflow.add_node("understand_goal", understand_goal)
    workflow.add_node("validate_alignment", validate_alignment)
    workflow.add_node("compute_score", compute_score)
    workflow.add_node("decide_outcome", decide_outcome)
    workflow.add_node("generate_agent_answer", generate_agent_answer)
    workflow.add_node("build_response", build_response)
    
    # Define edges
    workflow.set_entry_point("load_dataset")
    
    # Conditional edge from load_dataset
    def check_load_error(state):
        if state.get("error"):
            return "error"
        return "continue"
    
    workflow.add_conditional_edges(
        "load_dataset",
        check_load_error,
        {
            "error": END,
            "continue": "profile_dataset"
        }
    )
    
    workflow.add_edge("profile_dataset", "understand_goal")
    workflow.add_edge("understand_goal", "validate_alignment")
    workflow.add_edge("validate_alignment", "compute_score")
    workflow.add_edge("compute_score", "decide_outcome")
    workflow.add_edge("decide_outcome", "generate_agent_answer")
    workflow.add_edge("generate_agent_answer", "build_response")
    workflow.add_edge("build_response", END)
    
    return workflow.compile()

# Create singleton instance
validation_graph = create_validation_graph()