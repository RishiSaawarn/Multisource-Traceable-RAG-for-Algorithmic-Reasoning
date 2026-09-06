# scripts/verify_setup.py
"""Run this after setup to confirm everything works."""
import os
import sys
# Force UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()


def verify():
    checks = []

    # Check API keys
    for key in ["GROQ_API_KEY", "COHERE_API_KEY"]:
        val = os.getenv(key, "")
        status = "OK" if val and len(val) > 10 else "MISSING"
        mark = "[+]" if status == "OK" else "[!]"
        checks.append(f"{mark} {key}: {status}")

    # Check imports
    try:
        import langchain, chromadb, groq, cohere, rich  # noqa: F401
        checks.append("[+] Core packages installed")
    except ImportError as e:
        checks.append(f"[!] Missing package: {e}")

    # Test Groq connection
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say: SETUP OK"}],
            max_tokens=10
        )
        checks.append(f"[+] Groq API: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        checks.append(f"[!] Groq API: {e}")

    # Test Cohere embeddings
    try:
        import cohere
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
        resp = co.embed(
            texts=["test"],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )
        checks.append(f"[+] Cohere Embeddings: dim={len(resp.embeddings.float[0])}")
    except Exception as e:
        checks.append(f"[!] Cohere API: {e}")

    print("\n=== AlgoRAG Setup Verification ===")
    for check in checks:
        print(check)

    failed = [c for c in checks if c.startswith("[!]")]
    if not failed:
        print("\nAll checks passed. Run the ingestion pipeline next:")
        print("  python scripts/ingest.py")
    else:
        print(f"\n{len(failed)} check(s) failed. Fix the issues above before ingesting.")


if __name__ == "__main__":
    verify()