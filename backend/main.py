"""
main.py
-------
FastAPI backend for the AI-Powered Internship Recommendation System.

Endpoints
---------
GET  /                     → Serve the frontend HTML
GET  /api/health           → Health check
GET  /api/metadata         → Domain & location options for filter dropdowns
POST /api/recommend        → Get internship recommendations (JSON body)
POST /api/upload_resume    → Upload a PDF resume; returns parsed data
"""

import os
import sys

# Ensure project root is on the Python path so sub-modules resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional

from modules.recommender import RecommendationEngine
from modules.resume_parser import parse_resume

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Internship Recommendation API",
    description="ML-powered internship matching using TF-IDF + cosine similarity",
    version="1.0.0",
)

# Allow all origins during development (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS, JS) from the frontend directory
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
_STATIC_DIR = os.path.join(_FRONTEND_DIR, "static")

if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

# ---------------------------------------------------------------------------
# Initialise recommendation engine once at startup
# ---------------------------------------------------------------------------

engine = RecommendationEngine()


@app.on_event("startup")
def startup_event():
    # Set force_retrain=True to process the new 200-item structure
    engine.initialize(force_retrain=True)
    print("[API] Engine successfully indexed 200 internships.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    skills: str = Field(default="", description="Comma-separated or free-form skills")
    interests: str = Field(default="", description="Interests or preferred domains")
    resume_text: str = Field(default="", description="Paste plain-text resume (optional)")
    location: str = Field(default="", description="Preferred location filter")
    domain: str = Field(default="", description="Domain filter (e.g. AI/ML)")
    top_k: int = Field(default=8, ge=1, le=15, description="Number of results to return")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serve the main HTML page."""
    html_path = os.path.join(_FRONTEND_DIR, "templates", "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return JSONResponse({"message": "Frontend not found. API is running."})


@app.get("/api/health")
def health_check():
    """Simple liveness probe."""
    return {"status": "ok", "engine_ready": engine._ready}


@app.get("/api/metadata")
def get_metadata():
    """Return available filter options for the frontend dropdowns."""
    return {
        "domains": engine.get_all_domains(),
        "locations": engine.get_all_locations(),
    }


@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    """
    Main recommendation endpoint.

    Accepts user profile data and returns a ranked list of internship matches.
    """
    if not any([request.skills, request.interests, request.resume_text]):
        raise HTTPException(
            status_code=400,
            detail="Please provide at least one of: skills, interests, or resume_text.",
        )

    try:
        results = engine.recommend(
            skills=request.skills,
            interests=request.interests,
            resume_text=request.resume_text,
            location=request.location,
            domain=request.domain,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

    return {
        "count": len(results),
        "recommendations": results,
        "profile_summary": {
            "skills_provided": bool(request.skills),
            "interests_provided": bool(request.interests),
            "resume_uploaded": bool(request.resume_text),
            "filters": {
                "location": request.location or None,
                "domain": request.domain or None,
            },
        },
    }


@app.post("/api/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Accept a PDF resume, extract text, parse sections, and identify skills.

    Returns parsed resume data including extracted skills which can then
    be used to pre-fill the recommendation form.
    """
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Read file bytes (limit to 10 MB)
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    parsed = parse_resume(content)

    if parsed.get("error"):
        raise HTTPException(status_code=422, detail=parsed["error"])

    return {
        "filename": file.filename,
        "word_count": parsed["word_count"],
        "extracted_skills": parsed["skills"],
        "contact": parsed["contact"],
        "sections": {k: v[:500] for k, v in parsed.get("sections", {}).items()},  # truncate for response
        "raw_text_preview": parsed["raw_text"][:800],  # first 800 chars for display
    }


# ---------------------------------------------------------------------------
# Entry point (for running directly with `python main.py`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
