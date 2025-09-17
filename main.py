# main.py
from dotenv import load_dotenv
load_dotenv()
import os
import json
import re
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

# Optional: allow calls from frontend (React). Change origins for production.
from fastapi.middleware.cors import CORSMiddleware

#Load artifacts (existing) 
VECTOR_PATH = os.environ.get("VECTOR_PATH", "tfidf_vectorizer.joblib")
MATRIX_PATH = os.environ.get("MATRIX_PATH", "careers_tfidf.npz")
META_PATH = os.environ.get("META_PATH", "careers_meta.csv")

try:
    vectorizer = joblib.load(VECTOR_PATH)

    npz_file = np.load(MATRIX_PATH)
    tfidf_matrix = csr_matrix(
        (npz_file["data"], npz_file["indices"], npz_file["indptr"]),
        shape=tuple(npz_file["shape"])
    )

    careers_meta = pd.read_csv(META_PATH)

except Exception as e:
    print("ERROR LOADING FILES:", e)
    vectorizer, tfidf_matrix, careers_meta = None, None, None

# FastAPI setup 
app = FastAPI(title="Career Recommendation + Roadmap API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Skill extraction keywords 
SKILL_KEYWORDS = [
    "python", "sql", "excel", "statistics", "pandas", "numpy", "tableau", "powerbi", "r", "java",
    "c++", "javascript", "html", "css", "react", "node.js", "node", "django", "flask",
    "machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "aws", "gcp",
    "azure", "docker", "kubernetes", "git", "linux", "data visualization", "scikit-learn",
    "spark", "hadoop", "communication", "teamwork", "problem solving"
]

def extract_skills_from_text(text: str):
    if not text:
        return []
    t = text.lower()
    found = [kw for kw in SKILL_KEYWORDS if kw in t]
    return sorted(list(set(found)))

#  Gemini / LLM setup (optional) 
USE_GEMINI = False
try:
    import google.generativeai as genai  # will fail if not installed
    GEMINI_KEY = os.getenv("GEMINI_API_KEY", None)
    if GEMINI_KEY:
        genai.configure(api_key=GEMINI_KEY)
        USE_GEMINI = True
        print("Gemini configured.")
    else:
        print("GEMINI_API_KEY not provided; using local fallback generator.")
except Exception as e:
    print("google-generativeai not installed or import error:", e)
    USE_GEMINI = False

# helper: normalize roadmap into standard structure 
def normalize_roadmap(raw):
    """
    Convert LLM or fallback outputs into canonical structure:
    {
      "Beginner": [{"skill": "...", "status": "pending"}, ...],
      "Intermediate": [...],
      "Advanced": [...],
      "Mini Projects": [{"project":"...", "status":"pending"}, ...],
      "Main Projects": [{"project":"...", "status":"pending"}, ...]
    }
    """
    if not isinstance(raw, dict):
        return {"raw_text": str(raw)}

    out = {}
    # Ensure stage lists are list of dicts with skill/status
    for stage in ["Beginner", "Intermediate", "Advanced"]:
        items = raw.get(stage, [])
        norm = []
        for it in items:
            if isinstance(it, dict):
                # accept {"skill":..., "status":...} or {"skill":...}
                skill = it.get("skill") or it.get("name") or ""
                status = it.get("status", "pending")
            else:
                skill = str(it)
                status = "pending"
            norm.append({"skill": skill, "status": status})
        out[stage] = norm

    # Projects - convert strings -> dicts
    for pstage in ["Mini Projects", "Main Projects"]:
        plist = raw.get(pstage, [])
        normp = []
        for p in plist:
            if isinstance(p, dict):
                proj = p.get("project") or p.get("name") or ""
                status = p.get("status", "pending")
            else:
                proj = str(p)
                status = "pending"
            normp.append({"project": proj, "status": status})
        out[pstage] = normp

    return out

# fallback local roadmap generator (if no LLM) 
def fallback_local_roadmap(career: str, skills: List[str]):
    # generic lists; career-specific tweaks for common keywords
    base = {
        "Beginner": ["Python", "SQL", "Excel", "Statistics", "Data Visualization", "Linux"],
        "Intermediate": ["Pandas", "NumPy", "Data Cleaning", "Scikit-Learn", "Visualization Tools"],
        "Advanced": ["Deep Learning", "NLP", "MLOps", "Big Data (Spark)", "Cloud ML"],
        "Mini Projects": ["Data Cleaning with Pandas", "Exploratory Data Analysis"],
        "Main Projects": ["End-to-end ML Project", "Production-ready Data Pipeline"]
    }
    career_l = career.lower()
    if "web" in career_l or "frontend" in career_l or "react" in career_l:
        base["Beginner"] = ["HTML", "CSS", "JavaScript", "Git", "Basic UI"]
        base["Intermediate"] = ["React", "State Management", "REST APIs", "CSS Frameworks"]
        base["Advanced"] = ["Next.js", "Performance Optimization", "Testing", "SSR"]
        base["Mini Projects"] = ["Personal Portfolio", "Simple To-Do App"]
        base["Main Projects"] = ["E-commerce SPA", "Real-time Chat App"]
    elif "cloud" in career_l or "devops" in career_l or "site reliability" in career_l:
        base["Beginner"] = ["Linux", "Networking Basics", "Shell Scripting", "Git", "Python Basics"]
        base["Intermediate"] = ["Docker", "CI/CD", "AWS/GCP Basics", "Monitoring"]
        base["Advanced"] = ["Kubernetes", "Infrastructure as Code", "SRE Practices"]
        base["Mini Projects"] = ["Dockerize a Flask App"]
        base["Main Projects"] = ["K8s Production Deployment", "CI/CD Pipeline"]
    elif "data" in career_l or "data scientist" in career_l or "data analyst" in career_l:
        # leave base as-is (already data-oriented)
        pass
    elif "ml" in career_l or "machine" in career_l:
        base["Beginner"] = ["Python", "Linear Algebra Basics", "Statistics", "Probability"]
        base["Intermediate"] = ["Pandas", "Scikit-Learn", "Model Evaluation", "Feature Engineering"]
        base["Advanced"] = ["Deep Learning", "NLP", "Computer Vision", "MLOps"]
        base["Mini Projects"] = ["Regression Project", "Classifier on small dataset"]
        base["Main Projects"] = ["NLP Pipeline", "Image Classifier (transfer learning)"]
    # convert to normalized structure and mark everything "pending"
    raw = {}
    for k, v in base.items():
        raw[k] = v
    normalized = normalize_roadmap(raw)
    # ensure known skills included somewhere (no auto-complete; just include known skills if missing)
    known = [s.lower() for s in skills or []]
    for stage in ["Beginner", "Intermediate", "Advanced"]:
        stage_skills = [s["skill"].lower() for s in normalized[stage]]
        for ks in known:
            if ks not in stage_skills:
                # place known skills in Beginner if not present
                normalized["Beginner"].insert(0, {"skill": ks, "status": "pending"})
    return normalized

# LLM-driven roadmap function 
def generate_roadmap_with_gemini(career: str, skills: List[str] = None, soft_skills: List[str] = None, interests: List[str] = None, resume_text: str = ""):
    result = {"roadmap": None, "llm_used": "fallback"}  # default

    if USE_GEMINI:
        prompt = f"""
You are a career mentor that outputs structured JSON only.

Career: {career}
Skills reported by user (from profile/resume/quiz): {', '.join(skills or [])}
Soft skills: {', '.join(soft_skills or [])}
Interests: {', '.join(interests or [])}

Produce a detailed career roadmap JSON with sections:
- Beginner (5-7 core fundamentals)
- Intermediate (5-7 applied tools/frameworks)
- Advanced (5-7 specialized/cutting-edge skills)
- Mini Projects (3-5 small practice projects)
- Main Projects (2-3 portfolio-level projects)

Rules:
1) Do NOT mark anything as "completed" — mark all skills/projects as "pending". The user will update completion status.
2) Return JSON only (no extra text).
3) Each Beginner/Intermediate/Advanced item should be an object with fields: skill, status.
4) Projects lists may be arrays of strings or objects; we'll normalize them later.
"""
        try:
            # First try with Gemini Pro
            model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
            response = model.generate_content(prompt)
            txt = response.text.strip()
            parsed = json.loads(txt)
            result["roadmap"] = normalize_roadmap(parsed)
            result["llm_used"] = "gemini-1.5-pro"
            return result
        except Exception as e:
            print("Gemini-1.5-pro quota hit. Retrying with gemini-1.5-flash...", e)

            try:
                # Retry with Gemini Flash (cheaper + lighter)
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(prompt)
                print("Gemini raw response (flash):", response.text)
                txt = response.text.strip()
                parsed = json.loads(txt)
                result["roadmap"] = normalize_roadmap(parsed)
                result["llm_used"] = "gemini-1.5-flash"
                return result
            except Exception as e2:
                print("Gemini flash also failed. Falling back:", e2)

    # Fallback
    result["roadmap"] = fallback_local_roadmap(career, skills)
    result["llm_used"] = "fallback"
    return result
# API models 
class QueryPayload(BaseModel):
    query: str
    top_n: int = 5

class RoadmapPayload(BaseModel):
    career: str
    user_id: str
    skills: Optional[List[str]] = []
    soft_skills: Optional[List[str]] = []
    interests: Optional[List[str]] = []
    resume_text: Optional[str] = ""

class UpdatePayload(BaseModel):
    user_id: str
    item_name: str   # skill or project name
    status: str      # "completed" or "pending"

# Endpoints 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectorizer_loaded": vectorizer is not None,
        "matrix_shape": tfidf_matrix.shape if tfidf_matrix is not None else None,
        "gemini_configured": USE_GEMINI,
        "persistence_enabled": PERSIST
    }

@app.post("/recommend")
def recommend(payload: QueryPayload):
    if vectorizer is None or tfidf_matrix is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    query_vec = vectorizer.transform([payload.query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_n = min(payload.top_n, len(similarities))
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        row = careers_meta.iloc[idx].to_dict()
        row["score"] = float(similarities[idx])
        results.append(row)

    extracted_skills = extract_skills_from_text(payload.query)

    top_career = results[0].get("Career Name") if results else None
    roadmap_result = {"roadmap": None, "llm_used": "fallback"}
    if top_career:
        roadmap_result = generate_roadmap_with_gemini(top_career, skills=extracted_skills)

    return {
        "results": results,
        "extracted_skills": extracted_skills,
        "top_career_roadmap": roadmap_result["roadmap"],
        "llm_used": roadmap_result["llm_used"]
    }
@app.post("/roadmap")
def create_and_store_roadmap(payload: RoadmapPayload):
    # Generate roadmap (all pending)
    roadmap = generate_roadmap_with_gemini(payload.career, skills=payload.skills, soft_skills=payload.soft_skills, interests=payload.interests, resume_text=payload.resume_text)
    # store it under user_id
    user_roadmaps[payload.user_id] = {
        "career": payload.career,
        "roadmap": roadmap
    }
    save_user_roadmaps()
    return {"user_id": payload.user_id, "career": payload.career, "roadmap": roadmap}

@app.get("/user_roadmap/{user_id}")
def get_user_roadmap(user_id: str):
    if user_id not in user_roadmaps:
        raise HTTPException(status_code=404, detail="No roadmap for this user")
    return user_roadmaps[user_id]

@app.post("/update_progress")
def update_progress(payload: UpdatePayload):
    uid = payload.user_id
    if uid not in user_roadmaps:
        raise HTTPException(status_code=404, detail="User roadmap not found")

    roadmap = user_roadmaps[uid]["roadmap"]
    item = payload.item_name.strip().lower()
    updated = False

    # search skills in Beginner/Intermediate/Advanced
    for stage in ["Beginner", "Intermediate", "Advanced"]:
        for it in roadmap.get(stage, []):
            if it.get("skill","").strip().lower() == item:
                it["status"] = payload.status
                updated = True
                break
        if updated:
            break

    # search projects
    if not updated:
        for pstage in ["Mini Projects", "Main Projects"]:
            for it in roadmap.get(pstage, []):
                # it is dict {"project":..., "status":...}
                if it.get("project","").strip().lower() == item:
                    it["status"] = payload.status
                    updated = True
                    break
            if updated:
                break

    if not updated:
        raise HTTPException(status_code=404, detail="Item not found in user's roadmap")

    save_user_roadmaps()
    return {"message": f"Updated '{payload.item_name}' to '{payload.status}' for user {uid}"}

@app.get("/progress/{user_id}")
def get_progress(user_id: str):
    if user_id not in user_roadmaps:
        raise HTTPException(status_code=404, detail="User roadmap not found")
    roadmap = user_roadmaps[user_id]["roadmap"]
    summary = {}
    total_items = 0
    total_completed = 0
    for stage in ["Beginner", "Intermediate", "Advanced"]:
        items = roadmap.get(stage, [])
        count = len(items)
        completed = sum(1 for it in items if it.get("status") == "completed")
        summary[stage] = {"count": count, "completed": completed, "percent": round((completed/count*100) if count else 0,2)}
        total_items += count
        total_completed += completed

    # projects
    for pstage in ["Mini Projects", "Main Projects"]:
        pitems = roadmap.get(pstage, [])
        count = len(pitems)
        completed = sum(1 for it in pitems if it.get("status") == "completed")
        summary[pstage] = {"count": count, "completed": completed, "percent": round((completed/count*100) if count else 0,2)}
        total_items += count
        total_completed += completed

    overall_percent = round((total_completed / total_items * 100) if total_items else 0,2)
    return {"summary": summary, "overall_percent": overall_percent, "total_items": total_items, "total_completed": total_completed}
