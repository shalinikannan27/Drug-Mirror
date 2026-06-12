"""
DrugMirror – FastAPI Application Entry Point
============================================
Run with:
    uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Lifespan – startup logging
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Print startup diagnostics so developers know which mode is active."""
    import predict  # imported here so module-level prints fire first

    mode = "REAL" if predict.MODEL_LOADED else "MOCK"
    print(
        f"\n[DrugMirror] Server started in {mode} prediction mode.\n"
        f"[DrugMirror] Active labels ({len(predict.label_cols)}): "
        f"{predict.label_cols}\n"
    )
    yield  # application runs here


# ---------------------------------------------------------------------------
# App initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DrugMirror API",
    description=(
        "Backend for the DrugMirror drug side-effect prediction platform. "
        "Automatically switches between MOCK and REAL prediction modes "
        "depending on whether the trained model files exist in backend/models/."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS_ENV = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,https://drugmirror.vercel.app"
)
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS_ENV.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    smiles: str


class DDIRequest(BaseModel):
    smiles1: str
    smiles2: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health() -> dict:
    """Return service health and current prediction mode.

    Returns
    -------
    dict
        ``{"status": "ok", "model_loaded": bool, "mode": "real" | "mock"}``
    """
    import predict
    return {
        "status":       "ok",
        "model_loaded": predict.MODEL_LOADED,
        "mode":         "real" if predict.MODEL_LOADED else "mock",
    }


@app.post("/predict", tags=["Prediction"])
def predict_endpoint(body: PredictRequest) -> dict:
    """Predict drug side-effect probabilities from a SMILES string.

    Validates the SMILES, runs the full prediction pipeline, and returns
    molecular descriptors plus SHAP feature importances.

    Parameters
    ----------
    body : PredictRequest
        JSON body with ``smiles`` field.

    Returns
    -------
    dict
        Combined prediction + SHAP result with ``mode`` flag.

    Raises
    ------
    HTTPException 400
        If the SMILES string is invalid.
    HTTPException 500
        If an unexpected error occurs during prediction.
    """
    import predict
    import shap_explain
    from features import validate_smiles

    if not validate_smiles(body.smiles):
        raise HTTPException(
            status_code=400,
            detail={"valid": False, "error": "Invalid SMILES string"},
        )

    try:
        prediction_result = predict.predict_side_effects(body.smiles)
        shap_result       = shap_explain.get_shap_features(body.smiles, top_n=10)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"valid": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"Prediction failed: {exc}"})

    return {
        "valid":            True,
        "molecular_weight": prediction_result["molecular_weight"],
        "logp":             prediction_result["logp"],
        "qed":              prediction_result["qed"],
        "predictions":      prediction_result["predictions"],
        "shap_features":    shap_result,
        "mode":             "real" if predict.MODEL_LOADED else "mock",
    }


@app.get("/molecule", tags=["Visualisation"])
def molecule_endpoint(smiles: str = Query(..., description="URL-encoded SMILES string")) -> dict:
    """Generate a 2D molecule depiction and return it as base64 PNG.

    Parameters
    ----------
    smiles : str
        Query parameter; URL-encode the SMILES before sending.

    Returns
    -------
    dict
        ``{"image_base64": str}``

    Raises
    ------
    HTTPException 400
        If the SMILES is invalid or the image cannot be generated.
    """
    from molecule import get_molecule_image

    image_b64 = get_molecule_image(smiles)
    if image_b64 is None:
        raise HTTPException(
            status_code=400,
            detail={"valid": False, "error": "Invalid SMILES or image generation failed"},
        )
    return {"image_base64": image_b64}


@app.post("/ddi", tags=["Drug-Drug Interaction"])
def ddi_endpoint(body: DDIRequest) -> dict:
    """Check for potential drug–drug interaction between two molecules.

    Parameters
    ----------
    body : DDIRequest
        JSON body with ``smiles1`` and ``smiles2`` fields.

    Returns
    -------
    dict
        ``{"tanimoto_similarity": float, "risk_level": str, "explanation": str}``

    Raises
    ------
    HTTPException 400
        If either SMILES is invalid.
    HTTPException 500
        If an unexpected error occurs.
    """
    from features import validate_smiles
    from ddi import check_interaction

    if not validate_smiles(body.smiles1):
        raise HTTPException(
            status_code=400,
            detail={"valid": False, "error": "Invalid SMILES for drug 1"},
        )
    if not validate_smiles(body.smiles2):
        raise HTTPException(
            status_code=400,
            detail={"valid": False, "error": "Invalid SMILES for drug 2"},
        )

    try:
        result = check_interaction(body.smiles1, body.smiles2)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"valid": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"DDI check failed: {exc}"})

    return result


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
