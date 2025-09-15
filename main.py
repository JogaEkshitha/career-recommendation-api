import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# --- Load artifacts ---
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

app = FastAPI(title="Career Recommendation API")

class QueryPayload(BaseModel):
    query: str
    top_n: int = 5

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vectorizer_loaded": vectorizer is not None,
        "matrix_shape": tfidf_matrix.shape if tfidf_matrix is not None else None
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

    return {"results": results}
