import sys
import json
import time
sys.path.insert(0, ".")

from src.pipeline.orchestrator import VoiceRAGPipeline

ACCEPTANCE_QUERIES = [
    "இந்தியாவில் எத்தனை மாவட்டங்கள் உள்ளன?",
    "இந்தியாவில் எத்தனை மாநிலங்கள் உள்ளன?",
    "இந்தியாவின் தலைநகரம் என்ன?",
    "இந்தியாவில் மக்கள் தொகை அதிகம் உள்ள மாநிலம் எது?",
    "இந்தியாவில் அதிகமாக பயிரிடப்படும் பயிர்கள் என்ன?"
]

def run_acceptance_tests():
    print("=" * 100)
    print("HH GOA 2026 TASK 2 — MANDATORY 5 TAMIL ACCEPTANCE QUERIES TEST")
    print("=" * 100)
    
    pipeline = VoiceRAGPipeline()
    print("[ACCEPTANCE TEST] Initializing Voice RAG Pipeline...")
    pipeline.initialize_system()
    
    # Warmup
    _ = pipeline.run_text_query("Warmup query", mode="fast")
    print("[ACCEPTANCE TEST] Pipeline warmed up. Executing 5 mandatory Tamil queries...\n")

    all_passed = True

    for i, q in enumerate(ACCEPTANCE_QUERIES, 1):
        print(f"--- QUERY {i}: {q} ---")
        t0 = time.perf_counter()
        res = pipeline.run_text_query(q, mode="fast")
        wall_ms = (time.perf_counter() - t0) * 1000.0

        top_sources = res.get("sources", [])
        top_evidence = top_sources[0]["text"] if top_sources else "N/A"
        ret_score = top_sources[0]["score"] if top_sources else 0.0

        ans = res.get("answer", "")
        ans_mode = res.get("answer_mode", "")
        grounded = res.get("is_grounded", False)
        status = res.get("status", "")
        lat = res.get("latencies_ms", {})
        embed_ms = lat.get("embedding_ms", 0.0)
        faiss_ms = lat.get("dense_search_ms", 0.0)
        gen_ms = lat.get("generation_ms", 0.0)
        total_ms = lat.get("total_latency_ms", round(wall_ms, 2))

        print(f"Query:               {q}")
        print(f"Top Retrieved Evid:  {top_evidence[:120]}...")
        print(f"Retrieval Score:     {ret_score}")
        print(f"Answer:              {ans}")
        print(f"Answer Mode:         {ans_mode}")
        print(f"Grounded:            {grounded} (Status: {status})")
        print(f"Embedding Latency:   {embed_ms} ms")
        print(f"FAISS Search Lat:    {faiss_ms} ms")
        print(f"LLM Generation Lat:  {gen_ms} ms")
        print(f"Total Wall Latency:  {total_ms} ms")
        
        # Check assertions
        if q == "இந்தியாவில் எத்தனை மாவட்டங்கள் உள்ளன?":
            if "593" not in ans and "593" not in top_evidence:
                print("❌ FAIL: District query evidence missing 593")
                all_passed = False
            else:
                print("✅ PASS: District query returned valid 593 evidence/answer")

        if "unavailable" in ans.lower() or "மன்னிக்கவும்" in ans:
            if grounded is True:
                print("❌ FAIL: Refusal answer marked as Grounded = YES!")
                all_passed = False
            else:
                print("✅ PASS: Refusal answer correctly marked as Grounded = NO")
        
        if total_ms > 200:
            print(f"❌ FAIL: Latency {total_ms} ms exceeded 200 ms target")
            all_passed = False
        else:
            print(f"✅ PASS: Latency {total_ms} ms under 200 ms target")
            
        print("-" * 100 + "\n")

    if all_passed:
        print("🎉 ALL 5 MANDATORY ACCEPTANCE QUERIES PASSED PERFECTLY!")
    else:
        print("⚠️ SOME ACCEPTANCE CHECKS FAILED - SEE LOGS ABOVE.")

if __name__ == "__main__":
    run_acceptance_tests()
