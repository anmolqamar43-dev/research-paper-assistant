import json
from pathlib import Path

import faiss
import numpy as np
from google import genai


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Vercel serverless filesystem
# /tmp is writable but temporary
VECTOR_DIR = Path("/tmp/vector_data")
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = VECTOR_DIR / "research_papers.index"
METADATA_FILE = VECTOR_DIR / "metadata.json"


# --------------------------------------------------
# Create Gemini Client
# --------------------------------------------------

def create_gemini_client(api_key: str):
    return genai.Client(
        api_key=api_key
    )


# --------------------------------------------------
# Generate Embedding
# --------------------------------------------------

def generate_embedding(client, text: str):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(
        response.embeddings[0].values,
        dtype="float32"
    )


# --------------------------------------------------
# Build FAISS Vector Store
# --------------------------------------------------

def build_vector_store(client, chunks):

    if not chunks:
        raise ValueError(
            "No document chunks available."
        )

    vectors = []

    for chunk in chunks:

        embedding = generate_embedding(
            client,
            chunk["text"]
        )

        vectors.append(
            embedding
        )

    # Convert embeddings into NumPy matrix
    matrix = np.array(
        vectors,
        dtype="float32"
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(
        matrix
    )

    # Get embedding dimension
    dimension = matrix.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(
        dimension
    )

    # Add embeddings
    index.add(
        matrix
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save chunk metadata
    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    return {
        "chunks": len(chunks),
        "dimension": dimension
    }


# --------------------------------------------------
# Search Relevant Documents
# --------------------------------------------------

def search_documents(
    client,
    question: str,
    top_k: int = 5
):

    # Check if vector database exists
    if not INDEX_FILE.exists():

        raise FileNotFoundError(
            "No vector database exists. "
            "Upload a paper first."
        )

    # Load FAISS index
    index = faiss.read_index(
        str(INDEX_FILE)
    )

    # Load metadata
    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(file)

    # Create embedding for user's question
    question_vector = generate_embedding(
        client,
        question
    )

    # Reshape for FAISS
    question_vector = question_vector.reshape(
        1,
        -1
    )

    # Normalize question vector
    faiss.normalize_L2(
        question_vector
    )

    # Search similar chunks
    scores, positions = index.search(
        question_vector,
        min(top_k, index.ntotal)
    )

    results = []

    for score, position in zip(
        scores[0],
        positions[0]
    ):

        if position == -1:
            continue

        result = metadata[position].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results