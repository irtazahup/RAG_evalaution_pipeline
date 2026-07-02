import os
import time
import uuid

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone

from config import settings
from local_db import add_document

load_dotenv()

# 🔑 ENV
PINECONE_API_KEY = settings.PINECONE_API_KEY
INDEX_NAME = settings.INDEX_NAME
HF_TOKEN = settings.HUGGINGFACE_API_KEY
print(f"Using Pinecone Index: {INDEX_NAME}")
print(f"Using HuggingFace API Key: {HF_TOKEN[:4]}...{HF_TOKEN[-4:]}")  # Print only first and last 4 chars for security
print(f"pinecone_api_key: {PINECONE_API_KEY[:4]}...{PINECONE_API_KEY[-4:]}")  # Print only first and last 4 chars for security
# 🔌 Clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

hf_client = InferenceClient(api_key=HF_TOKEN)

MAX_RETRIES = 2


def _to_serializable(value):
    if hasattr(value, "tolist"):
        value = value.tolist()

    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            value = value.item()
        except Exception:
            pass

    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if isinstance(value, float):
        return float(value)

    if isinstance(value, int):
        return int(value)

    if isinstance(value, bool):
        return bool(value)

    return value


def _normalize_embedding(raw_embedding):
    if isinstance(raw_embedding, list) and raw_embedding and isinstance(raw_embedding[0], (list, tuple)):
        raw_embedding = raw_embedding[0]

    return _to_serializable(raw_embedding)


def _delete_document_vectors(document_id, vector_ids):
    if not vector_ids:
        return

    try:
        index.delete(ids=vector_ids)
        print(f"Deleted {len(vector_ids)} vectors for document {document_id}")
    except Exception as cleanup_error:
        print(f"Failed to clean up vectors for document {document_id}: {cleanup_error}")


def ingest_pdf(file_path: str):
    document_id = str(uuid.uuid4())
    timestamp = time.time()
    filename = os.path.basename(file_path)

    # ✅ Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # ✅ Chunking (≈500 tokens)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
    )

    docs = splitter.split_documents(pages)
    total_chunks = len(docs)

    texts = []
    metadatas = []
    for i, doc in enumerate(docs):
        texts.append(doc.page_content)
        metadatas.append({
            "text": doc.page_content,
            "source": filename,
            "chunk_index": i,
            "page": doc.metadata.get("page", None),
            "document_id": document_id,
            "timestamp": timestamp,
        })

    vector_ids = []

    try:
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            chunk_id = f"{document_id}_{i}"
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    embedding_response = hf_client.feature_extraction(
                        [text],
                        model="sentence-transformers/all-MiniLM-L6-v2",
                    )
                    embedding = _normalize_embedding(embedding_response)

                    vector = {
                        "id": chunk_id,
                        "values": embedding,
                        "metadata": _to_serializable(meta),
                    }
                    index.upsert(vectors=[vector])
                    vector_ids.append(chunk_id)
                    print(f"Stored chunk {i + 1}/{total_chunks} for {filename}")
                    break

                except Exception as exc:
                    last_error = exc
                    if attempt < MAX_RETRIES - 1:
                        print(
                            f"Retrying chunk {i + 1}/{total_chunks} for {filename} after error: {exc}"
                        )
                        time.sleep(1)
                        continue

                    raise RuntimeError(
                        f"Failed to store chunk {i + 1}/{total_chunks} for {filename} after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

            if last_error and not any(existing_id == chunk_id for existing_id in vector_ids):
                raise RuntimeError(
                    f"Chunk {i + 1}/{total_chunks} was not stored successfully for {filename}."
                )

    except Exception as exc:
        _delete_document_vectors(document_id, vector_ids)
        raise RuntimeError(f"Ingestion failed for {filename}: {exc}") from exc

    add_document(document_id, filename, total_chunks)
    return document_id, total_chunks