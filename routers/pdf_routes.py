import asyncio
import os
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from config import settings
from ingestion_pipeline import ingest_pdf

router = APIRouter(
    prefix="/pdf",
    tags=["PDF Ingestion"]
)


def _save_upload(file_path: str, upload_file) -> None:
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file, buffer)


@router.post("/ingest", status_code=status.HTTP_201_CREATED)
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to ingest a PDF file.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    try:
        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

        await asyncio.to_thread(_save_upload, file_path, file.file)
        print(f"File saved to {file_path}")
        # Ingest PDF
        document_id, chunk_count = await ingest_pdf(file_path)

        return {
            "document_id": document_id,
            "chunks_stored": chunk_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))