"""
DrugMirror – FastAPI Application Entry Point
============================================
Run with:
    cd backend/
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
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import predict
    mode = "REAL" if predict.MODEL_LOADED else "MOCK"
    print(
        f"\n[DrugMirror] Server started in {mode} prediction mode.\n"
        f"[DrugMirror] Active labels ({len(predict.label_cols)}): {predict.label_cols}\n"
    )
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DrugMirror API",
    description=(
        "Backend for the DrugMirror drug safety prediction platform. "
        "Automatically switches between REAL (trained models) and MOCK modes."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS_ENV = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://drugmirror.vercel.app,"
    "http://localhost:8000,http://127.0.0.1:8000,"
    "http://localhost:5500,http://127.0.0.1:5500,"
    "http://localhost:8080,http://127.0.0.1:8080,null"
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
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    smiles: str

class DDIRequest(BaseModel):
    smiles1: str
    smiles2: str

class AISummaryRequest(BaseModel):
    prompt: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
def health() -> dict:
    """Return service health and prediction mode."""
    import predict
    return {
        "status":       "ok",
        "model_loaded": predict.MODEL_LOADED,
        "mode":         "real" if predict.MODEL_LOADED else "mock",
    }


@app.get("/resolve-smiles/{drug_name}", tags=["Lookup"])
def resolve_smiles_endpoint(drug_name: str) -> dict:
    """Resolve a drug name to its SMILES string.

    Checks the local SMILES cache first, then falls back to PubChem.

    Returns
    -------
    dict  ``{"drug_name": str, "smiles": str | null, "source": "cache" | "pubchem" | "not_found"}``
    """
    import predict

    # Check local cache
    key = drug_name.strip().lower()
    if key in predict._smiles_cache:
        return {
            "drug_name": drug_name,
            "smiles":    predict._smiles_cache[key],
            "source":    "cache",
        }

    # PubChem fallback
    smiles = predict.resolve_smiles(drug_name)
    if smiles:
        return {"drug_name": drug_name, "smiles": smiles, "source": "pubchem"}

    return {"drug_name": drug_name, "smiles": None, "source": "not_found"}


@app.post("/predict", tags=["Prediction"])
def predict_endpoint(body: PredictRequest) -> dict:
    """Predict drug side-effect probabilities from a SMILES string."""
    import predict
    from features import validate_smiles

    if not validate_smiles(body.smiles):
        raise HTTPException(
            status_code=400,
            detail={"valid": False, "error": "Invalid SMILES string"},
        )

    try:
        result = predict.predict_side_effects(body.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"valid": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"Prediction failed: {exc}"})

    # Also run SHAP if available (non-critical — swallow errors)
    shap_result = []
    try:
        import shap_explain
        shap_result = shap_explain.get_shap_features(body.smiles, top_n=10)
    except Exception:
        pass

    return {
        "valid":            True,
        "molecular_weight": result["molecular_weight"],
        "logp":             result["logp"],
        "qed":              result["qed"],
        "predictions":      result["predictions"],
        "shap_features":    shap_result,
        "mode":             "real" if predict.MODEL_LOADED else "mock",
        "drug_name":        result["drug_name"],
        "binding_risk":     result["binding_risk"],
        "logbb":            result["logbb"],
        "cns_mpo":          result["cns_mpo"],
    }


@app.post("/ai-summary", tags=["AI Integration"])
def ai_summary_endpoint(body: AISummaryRequest) -> dict:
    """Call Claude API to generate a clinical summary, or use mock if no key."""
    import json
    import urllib.request
    import urllib.error

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # High-fidelity mock
        prompt_lower = body.prompt.lower()
        drug_name = "this molecule"
        if "completed for:" in body.prompt:
            try:
                drug_name = body.prompt.split("completed for:")[1].split("\n")[0].strip() or "this molecule"
            except Exception:
                pass

        cardiac_risk = "SAFE"
        liver_risk = "MODERATE"
        logbb_val = "-1.2"

        for marker, var_name in [("hERG (cardiac):", "cardiac_risk"), ("CYP3A4 (liver):", "liver_risk"), ("LogBB estimate:", "logbb_val")]:
            if marker.lower() in prompt_lower:
                try:
                    val = body.prompt.split(marker)[1].split("\n")[0].strip()
                    if var_name == "cardiac_risk": cardiac_risk = val
                    elif var_name == "liver_risk":  liver_risk = val
                    elif var_name == "logbb_val":   logbb_val = val
                except Exception:
                    pass

        try:
            cns_desc = (
                f"moderate to high likelihood of crossing the blood-brain barrier (logBB of {logbb_val})"
                if float(logbb_val) >= -1.0
                else f"low likelihood of central nervous system penetration based on its logBB estimate of {logbb_val}"
            )
        except Exception:
            cns_desc = "low likelihood of central nervous system penetration"

        text = (
            f"This analysis suggests a risk profile for {drug_name} characterized by significant potential "
            f"for gastrointestinal side effects. Cardiac and hepatic safety are assessed as {cardiac_risk.lower()} "
            f"and {liver_risk.lower()} respectively, based on hERG and CYP3A4 binding risk estimates. "
            f"The molecule exhibits {cns_desc}, suggesting a clean central nervous system safety profile."
        )
        return {"text": text}

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 200,
        "messages": [{"role": "user", "content": body.prompt}],
    }
    try:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10.0) as response:
            res_data = json.loads(response.read().decode())
            return {"text": res_data["content"][0]["text"]}
    except urllib.error.HTTPError as err:
        return {"text": f"Error: Anthropic API call failed ({err.code}). Fallback: This molecule exhibits potential GI side effects and moderate CYP3A4 risk, with low cardiac risk."}
    except Exception as exc:
        return {"text": f"Error: {exc}. Fallback: This molecule exhibits potential GI side effects and moderate CYP3A4 risk, with low cardiac risk."}


@app.get("/molecule", tags=["Visualisation"])
def molecule_endpoint(smiles: str = Query(..., description="URL-encoded SMILES string")) -> dict:
    """Generate a 2D molecule depiction and return it as base64 PNG."""
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
    """Check for potential drug-drug interaction between two molecules."""
    from features import validate_smiles
    from ddi import check_interaction

    if not validate_smiles(body.smiles1):
        raise HTTPException(status_code=400, detail={"valid": False, "error": "Invalid SMILES for drug 1"})
    if not validate_smiles(body.smiles2):
        raise HTTPException(status_code=400, detail={"valid": False, "error": "Invalid SMILES for drug 2"})

    try:
        result = check_interaction(body.smiles1, body.smiles2)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"valid": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"DDI check failed: {exc}"})

    return result


@app.post("/faers", tags=["FAERS Signal"])
def faers_endpoint(body: PredictRequest) -> dict:
    """Predict FAERS real-world adverse event signals from a SMILES string."""
    import predict
    from features import validate_smiles

    if not validate_smiles(body.smiles):
        raise HTTPException(status_code=400, detail={"valid": False, "error": "Invalid SMILES string"})

    try:
        faers_result = predict.predict_faers(body.smiles)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": f"FAERS analysis failed: {exc}"})

    return {
        "valid":      True,
        "drug_name":  faers_result["drug_name"],
        "signals":    faers_result["signals"],
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
