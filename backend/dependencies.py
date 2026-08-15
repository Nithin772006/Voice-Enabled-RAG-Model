import sys
from pathlib import Path

# Add project root to sys.path to resolve local imports cleanly when loaded as a module
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.workflow import build_rag_graph

# Compile the LangGraph instance once at server startup as a module-level cached singleton
_compiled_graph = None

def get_rag_graph():
    """Dependency injection provider returning the compiled LangGraph workflow singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_rag_graph()
    return _compiled_graph
