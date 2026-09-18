"""
ingest.py
Reads every .txt file in ./case_docs, splits each into overlapping chunks,
embeds them with Azure OpenAI, and upserts them into a Qdrant Cloud collection.

Run this once before starting the server, and again any time you edit the
documents in case_docs/.

Usage:
    python ingest.py
"""

import os
import glob
import uuid

from dotenv import load_dotenv
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

CASE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "case_docs")
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 120    # characters of overlap between consecutive chunks

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
QDRANT_COLLECTION = os.environ["QDRANT_COLLECTION"]


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Simple fixed-window character chunker with overlap."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap
    return chunks


def load_documents():
    """Returns a list of (doc_id, doc_title, full_text) for every .txt file."""
    docs = []
    paths = sorted(glob.glob(os.path.join(CASE_DOCS_DIR, "*.txt")))
    if not paths:
        raise RuntimeError(f"No .txt files found in {CASE_DOCS_DIR}")
    for path in paths:
        doc_id = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # First non-empty line as a human-readable title
        title = next((line.strip() for line in text.splitlines() if line.strip()), doc_id)
        docs.append((doc_id, title, text))
    return docs


def main():
    print("Loading documents from case_docs/ ...")
    docs = load_documents()
    print(f"Found {len(docs)} documents.")

    print("Connecting to Azure OpenAI and Qdrant ...")
    aoai = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Build all chunks first so we know the embedding dimension before creating the collection
    all_chunks = []  # list of dicts: doc_id, doc_title, chunk_index, text
    for doc_id, title, text in docs:
        pieces = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "doc_id": doc_id,
                "doc_title": title,
                "chunk_index": i,
                "text": piece,
            })
    print(f"Split into {len(all_chunks)} chunks total.")

    print("Requesting embeddings from Azure OpenAI ...")
    texts = [c["text"] for c in all_chunks]
    # Azure OpenAI embedding endpoint accepts batches; keep batches modest for reliability
    embeddings = []
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = aoai.embeddings.create(
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            input=batch,
        )
        embeddings.extend([item.embedding for item in resp.data])
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    vector_size = len(embeddings[0])
    print(f"Embedding dimension: {vector_size}")

    print(f"(Re)creating Qdrant collection '{QDRANT_COLLECTION}' ...")
    qdrant.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    print("Upserting points into Qdrant ...")
    points = []
    for chunk, vector in zip(all_chunks, embeddings):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "doc_id": chunk["doc_id"],
                "doc_title": chunk["doc_title"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            },
        ))
    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)

    print(f"Done. {len(points)} chunks embedded and stored in Qdrant collection '{QDRANT_COLLECTION}'.")


if __name__ == "__main__":
    main()
