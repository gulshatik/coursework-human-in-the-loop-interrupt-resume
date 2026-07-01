#!/usr/bin/env python3
"""
Human-in-the-loop (interrupt / resume) demo with LangGraph.

The graph contains a single node that triggers an interrupt asking the user
to confirm an action.  The interrupt payload is a dictionary with a question
and a list of allowed responses.  The stream loop detects the interrupt,
simulates a user response, and resumes the graph.  The final state contains
the chosen answer under the key `human_value`.
"""

from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# --------------------------------------------------------------------------- #
# 1. Graph state definition
# --------------------------------------------------------------------------- #
class GraphState(TypedDict, total=False):
    """State of the graph."""
    human_value: str          # value supplied by the user
    foo: str                  # arbitrary initial data


# --------------------------------------------------------------------------- #
# 2. Node that triggers an interrupt
# --------------------------------------------------------------------------- #
def interrupt_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    If the state does not yet contain a human_value, trigger an interrupt.
    After resumption, store the answer in the state.
    """
    # First visit: trigger interrupt
    if "human_value" not in state:
        # Preserve any existing data (e.g., foo) in the interrupt payload
        payload = {
            "type": "alert",
            "question": "Уверены, что хотите продолжить?",
            "allow_responds": ["approve", "reject"],
            "foo": state.get("foo", "initial data"),
        }
        interrupt(payload)
        # Execution stops here; the node will be called again after resume
        return state

    # After resume: the state is the payload dict
    # Store the answer in the final state
    answer = state.get("answer", "unknown")
    state["human_value"] = answer
    # Keep foo if present
    if "foo" not in state:
        state["foo"] = "initial data"
    return state


# --------------------------------------------------------------------------- #
# 3. Build the graph
# --------------------------------------------------------------------------- #
builder = StateGraph(GraphState)
builder.add_node("interrupt_node", interrupt_node)
builder.add_edge(START, "interrupt_node")
builder.set_entry_point("interrupt_node")

# Compile with an in-memory checkpoint
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# --------------------------------------------------------------------------- #
# 4. Stream processing loop
# --------------------------------------------------------------------------- #
def process_stream(stream):
    """
    Consume chunks from the stream.  When an interrupt is detected,
    simulate a user response and resume the graph.
    """
    for chunk in stream:
        # Print every chunk for visibility
        print(chunk)

        # Detect interrupt
        if "__interrupt__" in chunk:
            # The interrupt payload is stored in the first element's .value
            payload = chunk["__interrupt__"][0].value  # type: ignore
            # Simulate a user response (choose the first allowed option)
            simulated_answer = payload["allow_responds"][0]
            print(f"Simulated user answer: {simulated_answer}")

            # Add the answer to the payload and resume
            payload["answer"] = simulated_answer
            return Command(resume=payload)

    # No more chunks – stream finished
    return None


# --------------------------------------------------------------------------- #
# 5. Main execution
# --------------------------------------------------------------------------- #
def main():
    # Initial state with some arbitrary data
    initial_state: GraphState = {"foo": "initial data"}

    # Configuration for the thread (used for checkpointing)
    config = {"configurable": {"thread_id": "thread_1"}}

    # Start the stream
    stream = graph.stream(initial_state, config)

    while True:
        resume_cmd = process_stream(stream)
        if resume_cmd is None:
            # Stream finished
            print("\n--- Stream finished ---")
            break
        # Resume the graph with the simulated answer
        stream = graph.stream(resume_cmd, config)


if __name__ == "__main__":
    main()
