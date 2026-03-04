from langgraph.graph import StateGraph, END

from state import MLState
from planner import planner
from codegen import codegen
from executor_node import executor_node


def _route_after_planner(state: MLState) -> str:
    """If finished, go to END. Otherwise generate code for the new stage."""
    return END if state.get("finished") else "codegen"


def build_step_graph():
    """
    ONE-STEP graph: planner → codegen → executor → END
    The graph runs a single planner→codegen→executor cycle per invocation.
    Caller is responsible for persisting state between calls.
    """
    g = StateGraph(MLState)

    g.add_node("planner",      planner)
    g.add_node("codegen",      codegen)
    g.add_node("executor",     executor_node)

    g.set_entry_point("planner")

    # Planner either finishes or hands off to codegen
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"codegen": "codegen", END: END},
    )

    # Always run executor after codegen, then stop
    g.add_edge("codegen",  "executor")
    g.add_edge("executor", END)

    return g.compile()


# Compiled graph — imported by main.py
step_graph = build_step_graph()
