import logging
from fastapi import APIRouter, HTTPException
from retriever import run_rag
from pydantic import BaseModel


# --- Request/response models ---
class QueryRequest(BaseModel):
    index_path: str
    subject: str  # "physics" or "math"
    question: str
    k: int = 4


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query")
async def rag_query(req: QueryRequest):
    try:
        result = run_rag(req.index_path, req.subject, req.question, k=req.k)
        return result
    except FileNotFoundError as e:
        logging.error("RAG error: %s", e)
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logging.error("RAG error: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Unexpected RAG error", e)
        raise HTTPException(
            status_code=500, detail="Internal error during retrieval/generation"
        )
