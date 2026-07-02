from fastapi import FastAPI
from routers.pdf_routes import router as pdf_router
from config import settings
import os

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app = FastAPI(
    title="RAG Evaluation Pipeline",
    description="A FastAPI application for ingesting PDF documents and managing embeddings.",
    version="1.0.0"
)

app.include_router(pdf_router)
