from state import MLState


def planner(state: MLState) -> dict:
    """
    Deterministic planner — decides which stage to execute next.
    Returns only the fields that need updating.
    """
    if not state["dataset_loaded"]:
        return {"stage": "load_dataset"}

    if state["problem_type"] is None:
        return {"stage": "inspect_dataset"}

    if state["stage"] == "inspect_dataset":
        return {"stage": "train_baseline"}

    if state["stage"] == "train_baseline":
        return {"stage": "evaluate"}

    if state["stage"] == "evaluate":
        return {"finished": True}

    # Fallback — should not reach here in normal flow
    return {"finished": True}
