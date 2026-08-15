import sys
import os
import time
import subprocess
import logging
import pprint
import requests
from unittest.mock import patch
from pathlib import Path
from fastapi.testclient import TestClient

# Resolve import paths cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.main import app

logger = logging.getLogger("TestBackendSuite")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def print_banner(test_num: int, title: str) -> None:
    print("\n" + "=" * 80)
    print(f" TEST {test_num}: {title}")
    print("=" * 80)

def main() -> None:
    # Initialize FastAPI TestClient
    client = TestClient(app)
    test_results = {}
    
    # Pre-seed a valid mock WAV file so Test 9 (Audio Serving) works without running real TTS
    os.makedirs("data/audio_responses", exist_ok=True)
    mock_wav_path = "data/audio_responses/response_99999_tamil.wav"
    mock_wav_content = b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00@\x1f\x00\x00@\x1f\x00\x00\x01\x00\x08\x00data\x00\x00\x00\x00'
    with open(mock_wav_path, "wb") as f:
        f.write(mock_wav_content)

    # Initialize patch objects to protect tests 1-11 from external network calls
    mock_llm = patch(
        "llm.model.SarvamModel.generate", 
        return_value="வியாழன் சூரிய குடும்பத்தின் மிகப்பெரிய கோள் ஆகும். ஒரு நிறுவனம் என்பது சட்டம் சார்ந்த அமைப்பு. Machine learning and artificial intelligence definitions. [Mocked Answer]"
    )
    mock_tts = patch("tts.engine.TTSEngine.synthesize", return_value=mock_wav_path)
    
    logger.info("Starting mock patches for Unit/API tests...")
    mock_llm.start()
    mock_tts.start()
    
    # =================================================================
    # TEST 1: Health Endpoint (GET /health)
    # =================================================================
    print_banner(1, "Health Endpoint Liveness")
    try:
        res = client.get("/health")
        print(f"Status: {res.status_code} | Body: {res.json()}")
        is_success = res.status_code == 200 and res.json() == {"status": "ok"}
        test_results["Test 1 (Liveness)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 1 (Liveness)"] = f"FAIL (Crashed: {e})"
        
    # =================================================================
    # TEST 2: Health Readiness Endpoint (GET /health/ready)
    # =================================================================
    print_banner(2, "Health Endpoint Readiness")
    try:
        res = client.get("/health/ready")
        print(f"Status: {res.status_code} | Body: {res.json()}")
        is_success = res.status_code == 200 and res.json().get("status") == "ready"
        test_results["Test 2 (Readiness)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 2 (Readiness)"] = f"FAIL (Crashed: {e})"
        
    # =================================================================
    # TEST 3: Invalid Empty Query (POST /api/v1/rag/query)
    # =================================================================
    print_banner(3, "Invalid Empty Query Validation")
    try:
        res = client.post("/api/v1/rag/query", json={"query": "    "})
        print(f"Status: {res.status_code} | Body: {res.json()}")
        is_success = res.status_code == 422  # Pydantic validation error
        test_results["Test 3 (Empty Query)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 3 (Empty Query)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 4: Valid English Query
    # =================================================================
    print_banner(4, "Valid English Query")
    try:
        res = client.post("/api/v1/rag/query", json={"query": "What is machine learning?", "language": "english"})
        print(f"Status: {res.status_code}")
        body = res.json()
        pprint.pprint(body)
        is_success = (
            res.status_code == 200 and
            body.get("status") == "success" and
            body.get("audio_url") is not None
        )
        test_results["Test 4 (English Query)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 4 (English Query)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 5: Valid Tamil Query
    # =================================================================
    print_banner(5, "Valid Tamil Query")
    try:
        res = client.post("/api/v1/rag/query", json={"query": "ஒரு நிறுவனம் என்பது என்ன?", "language": "tamil"})
        print(f"Status: {res.status_code}")
        body = res.json()
        pprint.pprint(body)
        is_success = (
            res.status_code == 200 and
            body.get("status") == "success" and
            "நிறுவனம்" in body.get("answer", "") and
            body.get("audio_url") is not None
        )
        test_results["Test 5 (Tamil Query)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 5 (Tamil Query)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 6: Valid Telugu Query
    # =================================================================
    print_banner(6, "Valid Telugu Query")
    try:
        res = client.post("/api/v1/rag/query", json={"query": "మెషిన్ లెర్నింగ్ అంటే ఏమిటి?", "language": "telugu"})
        print(f"Status: {res.status_code}")
        body = res.json()
        pprint.pprint(body)
        is_success = (
            res.status_code == 200 and
            body.get("status") == "success" and
            body.get("audio_url") is not None
        )
        test_results["Test 6 (Telugu Query)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 6 (Telugu Query)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 7: Valid Hindi Query
    # =================================================================
    print_banner(7, "Valid Hindi Query")
    try:
        res = client.post("/api/v1/rag/query", json={"query": "मशीन लर्निंग क्या है?", "language": "hindi"})
        print(f"Status: {res.status_code}")
        body = res.json()
        pprint.pprint(body)
        is_success = (
            res.status_code == 200 and
            body.get("status") == "success" and
            body.get("audio_url") is not None
        )
        test_results["Test 7 (Hindi Query)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 7 (Hindi Query)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 8: Low-Confidence Query Fallback
    # =================================================================
    print_banner(8, "Low-Confidence Query Fallback")
    try:
        res = client.post("/api/v1/rag/query", json={"query": "What is photosynthesis?", "language": "english"})
        print(f"Status: {res.status_code}")
        body = res.json()
        pprint.pprint(body)
        is_success = (
            res.status_code == 200 and
            body.get("status") == "fallback" and
            "couldn't find enough" in body.get("answer", "").lower() and
            body.get("audio_url") is not None
        )
        test_results["Test 8 (Low-Confidence Fallback)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 8 (Low-Confidence Fallback)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 9: Audio Endpoint serving
    # =================================================================
    print_banner(9, "Audio Endpoint Serving")
    try:
        # Request a valid query first to ensure we have an audio filename
        res_query = client.post("/api/v1/rag/query", json={"query": "ஒரு நிறுவனம் என்பது என்ன?", "language": "tamil"})
        audio_url = res_query.json().get("audio_url")
        print(f"Retrieved Audio URL: {audio_url}")
        
        # Call the GET endpoint
        res_audio = client.get(audio_url)
        print(f"Audio Status: {res_audio.status_code} | Content-Type: {res_audio.headers.get('content-type')}")
        
        # Verify RIFF headers
        body_bytes = res_audio.content
        has_riff = body_bytes.startswith(b"RIFF")
        print(f"Body length: {len(body_bytes)} bytes | Valid WAV container (RIFF): {has_riff}")
        
        is_success = (
            res_audio.status_code == 200 and
            res_audio.headers.get("content-type") == "audio/wav" and
            len(body_bytes) > 0 and
            has_riff
        )
        test_results["Test 9 (Audio Serving)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 9 (Audio Serving)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 10: Missing Audio File (404)
    # =================================================================
    print_banner(10, "Missing Audio Endpoint (404)")
    try:
        res = client.get("/api/v1/audio/response_99999_english.wav")
        print(f"Status: {res.status_code} | Body: {res.json()}")
        is_success = res.status_code == 404
        test_results["Test 10 (Audio 404)"] = "PASS" if is_success else "FAIL"
    except Exception as e:
        test_results["Test 10 (Audio 404)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 11: Path Traversal Attack Blocked
    # =================================================================
    print_banner(11, "Path Traversal Blocking")
    try:
        malicious_paths = [
            "/api/v1/audio/../../etc/passwd",
            "/api/v1/audio/response_123_tamil.wav/..",
            "/api/v1/audio/abc.wav"  # Doesn't match response_\d+_[a-z]+\.wav regex format
        ]
        
        all_blocked = True
        for path in malicious_paths:
            res = client.get(path)
            print(f"Path: {path} | Status: {res.status_code}")
            if res.status_code not in [400, 404]:
                all_blocked = False
                
        test_results["Test 11 (Path Traversal Blocked)"] = "PASS" if all_blocked else "FAIL"
    except Exception as e:
        test_results["Test 11 (Path Traversal Blocked)"] = f"FAIL (Crashed: {e})"

    # =================================================================
    # TEST 12: REAL END-TO-END HTTP TEST (Subprocess Uvicorn launch)
    # =================================================================
    print_banner(12, "REAL End-to-End HTTP Subprocess Test")
    
    server_proc = None
    try:
        # Explicitly stop the mock patches to test the real RAG pipeline integration in the Uvicorn server
        logger.info("Stopping mock patches before E2E tests...")
        mock_llm.stop()
        mock_tts.stop()

        # Explicitly close the local Qdrant connections in this test runner process 
        # to release SQLite file locks before spawning the background Uvicorn server.
        try:
            from graph.nodes import get_retriever
            get_retriever().client.close()
            logger.info("Main process Qdrant client closed and database lock released.")
        except Exception as e:
            logger.warning(f"Failed to close local Qdrant client: {e}")

        python_exec = sys.executable
        logger.info("Starting Uvicorn server subprocess...")
        
        # Start uvicorn server in a separate subprocess (using DEVNULL to prevent pipe buffer writes deadlock)
        server_proc = subprocess.Popen(
            [python_exec, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8085"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait up to 10 seconds for health check to report OK
        url_health = "http://127.0.0.1:8085/health"
        is_ready = False
        for _ in range(20):
            try:
                res = requests.get(url_health, timeout=1.0)
                if res.status_code == 200:
                    is_ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.5)
            
        if not is_ready:
            raise RuntimeError("Uvicorn background server failed to report healthy in 10s.")
            
        logger.info("Uvicorn server is up and listening on port 8085. Sending query...")
        
        # Make a real HTTP post request (with a 60-second timeout to prevent hangs)
        query_payload = {"query": "ஒரு நிறுவனம் என்பது என்ன?", "language": "tamil"}
        res_query = requests.post("http://127.0.0.1:8085/api/v1/rag/query", json=query_payload, timeout=60.0)
        print(f"HTTP Post status: {res_query.status_code}")
        body = res_query.json()
        pprint.pprint(body)
        
        audio_url = body.get("audio_url")
        print(f"HTTP public audio URL: {audio_url}")
        
        # Fetch audio from public URL (with a 30-second timeout to prevent hangs)
        full_audio_url = f"http://127.0.0.1:8085{audio_url}"
        res_audio = requests.get(full_audio_url, timeout=30.0)
        print(f"HTTP GET audio status: {res_audio.status_code} | Content length: {len(res_audio.content)} bytes")
        
        has_riff = res_audio.content.startswith(b"RIFF")
        print(f"WAV RIFF Container verified: {has_riff}")
        
        is_success = (
            res_query.status_code == 200 and
            body.get("status") == "success" and
            audio_url is not None and
            res_audio.status_code == 200 and
            len(res_audio.content) > 0 and
            has_riff
        )
        test_results["Test 12 (Real HTTP E2E)"] = "PASS" if is_success else "FAIL"
        
    except Exception as e:
        logger.exception("Real HTTP E2E Test failed:")
        test_results["Test 12 (Real HTTP E2E)"] = f"FAIL (Crashed: {e})"
    finally:
        # Cleanly terminate uvicorn subprocess
        if server_proc:
            logger.info("Terminating Uvicorn subprocess server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=3.0)
                logger.info("Uvicorn subprocess shut down successfully.")
            except subprocess.TimeoutExpired:
                server_proc.kill()
                logger.info("Uvicorn subprocess killed force-fully.")

    # =================================================================
    # SUMMARY OF TESTS
    # =================================================================
    print("\n" + "=" * 80)
    print("                     PHASE 12 API BACKEND TEST SUMMARY")
    print("=" * 80)
    all_passed = True
    for t_name, t_res in test_results.items():
        print(f" {t_name:<45} ➔ {t_res}")
        if "PASS" not in t_res:
            all_passed = False
            
    print("=" * 80)
    if all_passed:
        print(" SUCCESS: All 12 API Backend tests completed with status: PASS")
        sys.exit(0)
    else:
        print(" FAILED: One or more backend integration tests failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
