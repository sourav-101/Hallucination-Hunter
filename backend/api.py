from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from backend.hallucination_hunter import HallucinationHunter, ClaimStatus

app = FastAPI(title="Hallucination Hunter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    source_documents: List[str]
    llm_output: str

def serialize_results(results):
    return {
        "trustScore": results["trust_score"],
        "summary": results["summary"],
        "claims": [
            {
                "id": r.claim.id,
                "text": r.claim.text,
                "status": r.status.value,
                "confidence": r.confidence,
                "explanation": r.explanation,
                "correction": r.correction,
                "evidence": [
                    {
                        "text": e.text,
                        "lineNumber": e.line_number,
                        "relevance": e.relevance_score,
                    }
                    for e in r.evidence
                ],
            }
            for r in results["results"]
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/verify")
def verify_document(payload: VerifyRequest):
    hunter = HallucinationHunter(payload.source_documents)
    results = hunter.verify_document(payload.llm_output)
    return serialize_results(results)
