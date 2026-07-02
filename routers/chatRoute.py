from fastapi import APIRouter
from scheme import ChatRequest
from utils import embeddings,retrieval,confidenceMeasure
from modules import chatFeature
router = APIRouter(
    prefix="/chat",
    tags=["Chatting"]
)

@router.post("")

async def chat(request:ChatRequest):
    """
    Endpoint to handle chat requests.
    """
    question = request.question
    # 1. Embed question
    vector = embeddings.get_embedding(request.question)

    # 2. Retrieve structured chunks
    chunks = retrieval.retrieve_context(vector)
    
     # 3. NOT FOUND HANDLING (VERY IMPORTANT)
    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
            "confidence": 0.0
        }
     # 4. Generate answer
    answer = chatFeature.generate_answer(request.question, chunks)

    # 5. Detect hallucination guard
    if "NOT_FOUND" in answer:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": chunks,
            "confidence": confidenceMeasure.compute_confidence(chunks)
        }

    # 6. Final response
    return {
        "answer": answer,
        "sources": chunks,
        "confidence": confidenceMeasure.compute_confidence(chunks)
    }


    

    



