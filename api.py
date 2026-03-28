"""
FastAPI REST endpoint for the Rural Health Triage engine.
Run:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from app import TriageEngine, format_result

app = FastAPI(
    title="Rural Health Triage API",
    description="Offline AI symptom triage — English + Hindi",
    version="1.0.0",
)

engine = TriageEngine()


class TriageRequest(BaseModel):
    symptoms: str = Field(..., description="Free-text symptoms (English or Hindi)")
    age: int = Field(30, ge=0, le=120, description="Patient age")
    gender: str = Field("Male", description="Male or Female")
    language: str = Field("en", description="'en' for English, 'hi' for Hindi")


class TriageResponse(BaseModel):
    urgency: str
    disease: str | None
    confidence: float
    detected_symptoms: list[str]
    advice: str
    explanation: str
    shap_text: str
    top3: list[dict]
    is_flag: bool
    formatted_markdown: str


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    """Run symptom triage and return structured result."""
    gender_int = 1 if req.gender.lower() in ("male", "पुरुष") else 0
    result = engine.predict(
        req.symptoms, age=req.age, gender=gender_int, lang=req.language
    )
    formatted = format_result(result, req.language)
    return TriageResponse(
        urgency=result["urgency"],
        disease=result["disease"],
        confidence=result["confidence"],
        detected_symptoms=result["detected"],
        advice=result["advice"],
        explanation=result["explanation"],
        shap_text=result.get("shap_text", ""),
        top3=result["top3"],
        is_flag=result["is_flag"],
        formatted_markdown=formatted,
    )


@app.get("/health")
def health_check():
    """Simple health-check endpoint."""
    return {"status": "ok", "model": "XGBoost triage", "offline": True}
