from fastapi import APIRouter,Depends,status,HTTPException, UploadFile ,File
from ingestion_pipeline import ingest_pdf
from config import settings
import os
import shutil

router = APIRouter(
    prefix="/pdf",
    tags=["PDF Ingestion"]
)

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

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved to {file_path}")
        # Ingest PDF
        document_id, chunk_count =  ingest_pdf(file_path)

        return {
            "document_id": document_id,
            "chunks_stored": chunk_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))