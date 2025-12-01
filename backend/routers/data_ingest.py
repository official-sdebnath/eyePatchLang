from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import tempfile
import logging
import uuid

from ingest import ingest_pdf

router = APIRouter()


@router.post("/ingest")
async def ingest_data(file: UploadFile = File(...)):
    """
    Ingest a PDF file -> split -> embed -> save FAISS index.
    Returns a unique job_id for tracking.
    """
    # 1) Validate
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # 2) Create a job ID
    job_id = uuid.uuid4().hex

    # 3) Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    logging.info(
        f"[{job_id}] Starting ingestion for {file.filename} -> temp: {tmp_path}"
    )

    try:
        # 4) Process the file
        index_path, num_chunks = ingest_pdf(tmp_path)

        logging.info(f"[{job_id}] Ingestion successful -> index: {index_path}")

        return {
            "job_id": job_id,
            "message": "Ingestion successful",
            "filename": file.filename,
            "index_path": index_path,
            "chunks": num_chunks,
        }

    except Exception as e:
        logging.error(f"[{job_id}] Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    finally:
        # 5) Clean temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
