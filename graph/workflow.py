import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Resolve local imports cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langgraph.graph import StateGraph, START, END
from graph.state import RAGGraphState
from graph.nodes import (
    input_processor_node,
    retrieval_node,
    guardrail_node,
    prompt_builder_node,
    llm_generation_node,
    fallback_node,
    tts_node
)

logger = logging.getLogger("RAGWorkflow")

# =================================================================
# CONDITIONAL ROUTING FUNCTIONS
# =================================================================

def route_after_input(state: RAGGraphState) -> str:
    """Routes state after input processing depending on input validity and scope."""
    status = state.get("status")
    if status == "error":
        logger.info("Routing after input: ERROR state detected. Routing directly to END.")
        return "END"
    elif status == "fallback":
        logger.info("Routing after input: OUT-OF-SCOPE query detected. Routing directly to FallbackNode.")
        return "fallback"
    else:
        logger.info("Routing after input: Query SAFE and IN-SCOPE. Routing to RetrievalNode.")
        return "retrieval"

def route_after_guardrail(state: RAGGraphState) -> str:
    """Routes state after retrieval guardrail check."""
    status = state.get("status")
    if status == "error":
        logger.info("Routing after guardrail: ERROR state detected. Routing to END.")
        return "END"
    elif state.get("retrieval_passed"):
        logger.info("Routing after guardrail: Retrieval passed similarity checks. Routing to PromptBuilderNode.")
        return "prompt_builder"
    else:
        logger.info("Routing after guardrail: Low similarity confidence or no chunks. Routing to FallbackNode.")
        return "fallback"

def route_after_generation(state: RAGGraphState) -> str:
    """Routes state after LLM generation depending on output safety checks."""
    status = state.get("status")
    if status == "error":
        logger.info("Routing after generation: ERROR state detected. Routing to END.")
        return "END"
    elif status == "fallback" or not state.get("answer_valid"):
        logger.info("Routing after generation: LLM generation was unsafe, leaked prompt, or failed. Routing to FallbackNode.")
        return "fallback"
    else:
        logger.info("Routing after generation: LLM generation valid and grounded. Routing to TTSNode.")
        return "tts"

# =================================================================
# GRAPH COMPILATION
# =================================================================

def build_rag_graph():
    """Assembles nodes, sequential edges, and conditional edges into a compiled LangGraph."""
    logger.info("Building RAG + Voice LangGraph workflow...")
    
    # Initialize state graph builder
    builder = StateGraph(RAGGraphState)
    
    # 1. Register logical nodes
    builder.add_node("input_processor", input_processor_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("prompt_builder", prompt_builder_node)
    builder.add_node("llm_generation", llm_generation_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("tts", tts_node)
    
    # 2. Add entry points and sequential edges
    builder.add_edge(START, "input_processor")
    builder.add_edge("retrieval", "guardrail")
    builder.add_edge("prompt_builder", "llm_generation")
    builder.add_edge("fallback", "tts")
    builder.add_edge("tts", END)
    
    # 3. Add conditional routing edges
    builder.add_conditional_edges(
        "input_processor",
        route_after_input,
        {
            "retrieval": "retrieval",
            "fallback": "fallback",
            "END": END
        }
    )
    
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "prompt_builder": "prompt_builder",
            "fallback": "fallback",
            "END": END
        }
    )
    
    builder.add_conditional_edges(
        "llm_generation",
        route_after_generation,
        {
            "tts": "tts",
            "fallback": "fallback",
            "END": END
        }
    )
    
    # 4. Compile and return compiled graph runner
    logger.info("Compiling LangGraph workflow...")
    return builder.compile()
