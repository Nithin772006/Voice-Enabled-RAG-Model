import sys
import os
import logging
import pprint
from unittest.mock import patch
from pathlib import Path

# Resolve import paths cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from graph.workflow import build_rag_graph
from graph.nodes import FALLBACK_RESPONSES

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TestGraphSuite")

def print_banner(test_num: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f" TEST {test_num}: {title}")
    print("=" * 80)

def main() -> None:
    # 1. Compile the LangGraph RAG workflow once
    graph = build_rag_graph()
    
    # Track test pass/fail results
    test_results = {}
    
    # =================================================================
    # TEST 1: Valid English Query (Real ML Retrieval & Grounding)
    # =================================================================
    print_banner(1, "Valid English Query (What is machine learning?)")
    try:
        state = graph.invoke({
            "query": "What is machine learning?",
            "language": "english"
        })
        pprint.pprint(state)
        
        # Verify TTS audio generation
        audio = state.get("audio_path")
        ans = state.get("answer", "").lower()
        has_keywords = any(kw in ans for kw in ["machine", "learning", "मशीन", "लर्निंग"])
        is_success = (
            state.get("status") == "success" and
            state.get("retrieval_passed") is True and
            has_keywords and
            audio is not None and os.path.exists(audio)
        )
        test_results["Test 1 (English Success)"] = "PASS" if is_success else f"FAIL (status: {state.get('status')}, audio: {audio})"
    except Exception as e:
        test_results["Test 1 (English Success)"] = f"FAIL (Crashed: {e})"
        
    # =================================================================
    # TEST 2: Valid Tamil Query (Real Company Retrieval & Grounding)
    # =================================================================
    print_banner(2, "Valid Tamil Query (ஒரு நிறுவனம் என்பது என்ன?)")
    try:
        state = graph.invoke({
            "query": "ஒரு நிறுவனம் என்பது என்ன?",
            "language": "tamil"
        })
        pprint.pprint(state)
        
        audio = state.get("audio_path")
        is_success = (
            state.get("status") == "success" and
            state.get("retrieval_passed") is True and
            "நிறுவனம்" in state.get("answer", "") and
            audio is not None and os.path.exists(audio)
        )
        test_results["Test 2 (Tamil Success)"] = "PASS" if is_success else f"FAIL (status: {state.get('status')}, audio: {audio})"
    except Exception as e:
        test_results["Test 2 (Tamil Success)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 3: Valid Telugu Query (Real ML Retrieval & Grounding)
    # =================================================================
    print_banner(3, "Valid Telugu Query (మెషిన్ లెర్నింగ్ అంటే ఏమిటి?)")
    try:
        state = graph.invoke({
            "query": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?",
            "language": "telugu"
        })
        pprint.pprint(state)
        
        audio = state.get("audio_path")
        is_success = (
            state.get("status") == "success" and
            state.get("retrieval_passed") is True and
            audio is not None and os.path.exists(audio)
        )
        test_results["Test 3 (Telugu Success)"] = "PASS" if is_success else f"FAIL (status: {state.get('status')}, audio: {audio})"
    except Exception as e:
        test_results["Test 3 (Telugu Success)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 4: Valid Hindi Query (Real ML Retrieval & Grounding)
    # =================================================================
    print_banner(4, "Valid Hindi Query (मशीन लर्निंग क्या है?)")
    try:
        state = graph.invoke({
            "query": "मशीन लर्निंग क्या है?",
            "language": "hindi"
        })
        pprint.pprint(state)
        
        audio = state.get("audio_path")
        is_success = (
            state.get("status") == "success" and
            state.get("retrieval_passed") is True and
            audio is not None and os.path.exists(audio)
        )
        test_results["Test 4 (Hindi Success)"] = "PASS" if is_success else f"FAIL (status: {state.get('status')}, audio: {audio})"
    except Exception as e:
        test_results["Test 4 (Hindi Success)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 5: Low-Confidence Query (Bypasses LLM, routes to Fallback)
    # =================================================================
    print_banner(5, "Low-Confidence Query (What is photosynthesis?)")
    try:
        state = graph.invoke({
            "query": "What is photosynthesis?",
            "language": "english"
        })
        pprint.pprint(state)
        
        # Verify that it triggered fallback and did NOT call PromptBuilder or LLM
        audio = state.get("audio_path")
        is_success = (
            state.get("status") == "fallback" and
            state.get("retrieval_passed") is False and
            state.get("prompt") is None and  # Prompt node bypassed
            audio is not None and os.path.exists(audio)  # Plays fallback sound
        )
        test_results["Test 5 (Low-Confidence Fallback)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 5 (Low-Confidence Fallback)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 6: Empty/Invalid Query Input (Syntax Validation Reject)
    # =================================================================
    print_banner(6, "Empty/Invalid Query Input")
    try:
        state = graph.invoke({
            "query": "   ",
            "language": "tamil"
        })
        pprint.pprint(state)
        
        is_success = (
            state.get("status") == "error" and
            "empty" in state.get("error", "").lower() and
            state.get("audio_path") is None  # No TTS generated
        )
        test_results["Test 6 (Invalid Input error)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 6 (Invalid Input error)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 7: Prompt Injection in Retrieved Context (Treated as Data)
    # =================================================================
    print_banner(7, "Prompt Injection Context Grounding Protection")
    try:
        # Query matching injection chunk (chunk_003)
        state = graph.invoke({
            "query": "இணைப்பு வெற்றிகரமாக முடிந்தது.",
            "language": "tamil"
        })
        pprint.pprint(state)
        
        # The prompt injection is inside the chunk text, not user query.
        # Check if the LLM followed injection or handled it as data / fallback.
        # Since it tells the model "Ignore previous instructions", the model should ignore it
        # and answer from context or trigger standard grounding rules.
        is_success = (
            state.get("status") in ["success", "fallback"] and
            "INJECTION" not in state.get("answer", "")  # Verification
        )
        test_results["Test 7 (Prompt Injection Protected)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 7 (Prompt Injection Protected)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 8: Simulated LLM Failure (Graceful Fallback Node Routing)
    # =================================================================
    print_banner(8, "Simulated LLM API Failure")
    # Patch SarvamModel.generate to raise a mock API connection crash
    with patch("llm.model.SarvamModel.generate", side_effect=RuntimeError("Sarvam API Unavailable (Simulated)")):
        try:
            state = graph.invoke({
                "query": "ஒரு நிறுவனம் என்பது என்ன?",
                "language": "tamil"
            })
            pprint.pprint(state)
            
            is_success = (
                state.get("status") == "fallback" and
                "Sarvam API Unavailable" in state.get("error", "") and
                state.get("answer") == FALLBACK_RESPONSES["tamil"]  # Default Tamil RAG fallback loaded
            )
            test_results["Test 8 (LLM Crash Fallback)"] = "PASS" if is_success else "FAIL"
        except Exception as e:
            test_results["Test 8 (LLM Crash Fallback)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 9: Simulated TTS Failure (Graceful audio_path=None Handling)
    # =================================================================
    print_banner(9, "Simulated TTS API Failure")
    # Patch TTSEngine.synthesize to raise a simulated file permission/connection crash
    with patch("tts.engine.TTSEngine.synthesize", side_effect=RuntimeError("TTS Speech Endpoint Timeout (Simulated)")):
        try:
            state = graph.invoke({
                "query": "ஒரு நிறுவனம் என்பது என்ன?",
                "language": "tamil"
            })
            pprint.pprint(state)
            
            is_success = (
                # Answer is generated successfully but audio_path remains None due to TTS failure
                state.get("answer_valid") is True and
                state.get("audio_path") is None and
                "TTS Speech Endpoint Timeout" in state.get("error", "")
            )
            test_results["Test 9 (TTS Crash Handling)"] = "PASS" if is_success else "FAIL"
        except Exception as e:
            test_results["Test 9 (TTS Crash Handling)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # SUMMARY OF TESTS
    # =================================================================
    print("\n" + "=" * 80)
    print("                     PHASE 11 STATE GRAPH TEST SUMMARY")
    print("=" * 80)
    all_passed = True
    for t_name, t_res in test_results.items():
        print(f" {t_name:<45} ➔ {t_res}")
        if "PASS" not in t_res:
            all_passed = False
            
    print("=" * 80)
    if all_passed:
        print(" SUCCESS: All 9 LangGraph integration tests completed with status: PASS")
        sys.exit(0)
    else:
        print(" FAILED: One or more integration tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
