"""
Build the policy semantic index using Gemini embeddings.

This script is run ONCE during development to pre-compute
the policy clause embeddings and save them as a portable JSON file.

The JSON file is committed to the repository so the production
app has zero startup embedding cost.

Usage:
    python scripts/build_index.py
"""

import os
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from google import genai

POLICY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "policy.txt"
)

OUTPUT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "policy_index.json"
)

EMBEDDING_MODEL = "gemini-embedding-001"


def chunk_policy(text: str) -> list[dict]:
    """Split policy into clause-level chunks."""
    chunks = []
    current_clause_id = ""
    current_text = ""

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("##"):
            # Save previous chunk
            if current_text.strip():
                chunks.append({
                    "clause_id": current_clause_id,
                    "text": current_text.strip(),
                })
            current_clause_id = stripped.replace("#", "").strip()
            current_text = ""
        elif stripped:
            # Check for sub-clause numbering (e.g. "4.1 ...")
            if len(stripped) > 3 and stripped[0].isdigit() and "." in stripped[:4]:
                if current_text.strip():
                    chunks.append({
                        "clause_id": current_clause_id,
                        "text": current_text.strip(),
                    })
                current_clause_id = stripped.split(" ")[0]
                current_text = stripped
            else:
                current_text += " " + stripped

    # Save last chunk
    if current_text.strip():
        chunks.append({
            "clause_id": current_clause_id,
            "text": current_text.strip(),
        })

    return chunks


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Read policy
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        policy_text = f.read()

    print(f"Loaded policy: {len(policy_text)} characters")

    # Chunk
    chunks = chunk_policy(policy_text)
    print(f"Split into {len(chunks)} chunks")

    for chunk in chunks:
        print(f"  [{chunk['clause_id']}] {chunk['text'][:80]}...")

    # Embed each chunk
    print("\nGenerating embeddings...")
    for i, chunk in enumerate(chunks):
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk["text"],
        )
        chunk["embedding"] = result.embeddings[0].values
        print(f"  Embedded chunk {i+1}/{len(chunks)}: {chunk['clause_id']}")

    # Save index
    index = {"chunks": chunks}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nPolicy index saved to: {OUTPUT_FILE}")
    print(f"Total chunks: {len(chunks)}")

    # Verification: test query
    print("\n--- Verification Test ---")
    test_query = "racing on a racetrack"
    test_result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=test_query,
    )
    import numpy as np
    query_vec = np.array(test_result.embeddings[0].values)

    best_score = -1
    best_chunk = None
    for chunk in chunks:
        chunk_vec = np.array(chunk["embedding"])
        score = float(np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)))
        if score > best_score:
            best_score = score
            best_chunk = chunk

    print(f"Query: '{test_query}'")
    print(f"Top match: [{best_chunk['clause_id']}] score={best_score:.4f}")
    print(f"Text: {best_chunk['text'][:120]}...")

    if "racing" in best_chunk["text"].lower() or "racetrack" in best_chunk["text"].lower():
        print("\n✅ VERIFICATION PASSED: Racing exclusion clause correctly retrieved.")
    else:
        print("\n⚠ VERIFICATION WARNING: Top result may not be the racing clause. Review manually.")


if __name__ == "__main__":
    main()
