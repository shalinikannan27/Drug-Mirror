"""
DrugMirror – Side-Effect Prediction
======================================
╔══════════════════════════════════════════════════════════════════════════╗
║  MOCK MODE NOTICE                                                        ║
║                                                                          ║
║  This file generates fake but DETERMINISTIC predictions for frontend     ║
║  development while the ML model is being trained.                        ║
║                                                                          ║
║  Once these three files are placed in backend/models/ the system will    ║
║  automatically switch to REAL predictions on next restart — NO code      ║
║  changes are needed (unless feature_cols ordering doesn't match the      ║
║  fallback concatenation order defined in features.py):                   ║
║                                                                          ║
║      drugmirror_models.pkl   – dict of {label: sklearn estimator}        ║
║      label_cols.pkl          – list of 12 side-effect label names        ║
║      feature_cols.pkl        – ordered list of training column names     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import hashlib

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Model file paths (from .env with sensible defaults)
# ---------------------------------------------------------------------------
MODEL_PATH       = os.getenv("MODEL_PATH",       "backend/models/drugmirror_models.pkl")
LABEL_COLS_PATH  = os.getenv("LABEL_COLS_PATH",  "backend/models/label_cols.pkl")
FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH", "backend/models/feature_cols.pkl")

# ---------------------------------------------------------------------------
# Hardcoded fallback labels (used when label_cols.pkl is absent)
# ---------------------------------------------------------------------------
FALLBACK_LABEL_COLS: list[str] = [
    "gi_issues",
    "skin_reactions",
    "neurological",
    "fatigue",
    "pain",
    "cardiovascular",
    "respiratory",
    "immune",
    "metabolic",
    "haematological",
    "psychiatric",
    "autonomic",
]

# ---------------------------------------------------------------------------
# Attempt to load model files
# ---------------------------------------------------------------------------
MODEL_LOADED: bool = False
models: dict = {}
label_cols: list[str] = FALLBACK_LABEL_COLS
feature_cols: list[str] | None = None

_all_files_present = all(
    os.path.exists(p) for p in [MODEL_PATH, LABEL_COLS_PATH, FEATURE_COLS_PATH]
)

if _all_files_present:
    try:
        import joblib

        models       = joblib.load(MODEL_PATH)
        label_cols   = joblib.load(LABEL_COLS_PATH)
        feature_cols = joblib.load(FEATURE_COLS_PATH)
        MODEL_LOADED = True
        print("[DrugMirror] [OK] ML model loaded successfully - running in REAL PREDICTION mode.")
        print(f"[DrugMirror] Labels ({len(label_cols)}): {label_cols}")
    except Exception as exc:
        print(f"[DrugMirror] [WARN] Model files found but failed to load: {exc}")
        print("[DrugMirror] Falling back to MOCK PREDICTION mode.")
else:
    print(
        "\n"
        "================================================================\n"
        "  WARNING: ML model not found, running in MOCK PREDICTION mode  \n"
        "  for frontend development.                                      \n"
        "================================================================\n"
    )
    print(f"[DrugMirror] Using fallback labels ({len(label_cols)}): {label_cols}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deterministic_prob(smiles: str, label: str) -> float:
    """Return a pseudo-random but deterministic probability in [0.2, 0.9].

    The same (smiles, label) pair always produces the same value, which is
    stable across restarts — useful for frontend testing without a trained
    model.

    Parameters
    ----------
    smiles : str
        Canonicalised SMILES string.
    label : str
        Side-effect category name.

    Returns
    -------
    float
        Probability in the range [0.2, 0.9].
    """
    seed_str = f"{smiles.strip()}::{label}"
    digest = hashlib.sha256(seed_str.encode()).hexdigest()
    # Take first 8 hex chars → integer in [0, 0xFFFFFFFF]
    int_val = int(digest[:8], 16)
    # Normalise to [0.0, 1.0] then remap to [0.2, 0.9]
    normalised = int_val / 0xFFFFFFFF
    return round(0.2 + normalised * 0.7, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_side_effects(smiles: str) -> dict:
    """Predict drug side-effect probabilities for a given SMILES string.

    Operates in one of two modes:

    * **REAL** (``MODEL_LOADED = True``): assembles the feature vector, runs
      each sklearn estimator's ``predict_proba``, and returns the positive-
      class probability rounded to 3 decimal places.

    * **MOCK** (``MODEL_LOADED = False``): generates deterministic fake
      probabilities derived from a SHA-256 hash of the SMILES + label name.
      The same SMILES always returns the same output — useful for UI work.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.

    Returns
    -------
    dict
        ``{``
        ``"molecular_weight": float,``
        ``"logp": float,``
        ``"qed": float,``
        ``"predictions": {label: probability, ...}``
        ``}``

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed by RDKit.
    """
    from features import extract_features, assemble_feature_vector  # local import

    extracted = extract_features(smiles)
    raw = extracted["_raw"]

    if MODEL_LOADED:
        # ── Real prediction pipeline ─────────────────────────────────────
        feature_vector = assemble_feature_vector(extracted, feature_cols)
        predictions: dict[str, float] = {}
        for label in label_cols:
            clf = models.get(label)
            if clf is None:
                predictions[label] = 0.0
                continue
            try:
                prob = clf.predict_proba(feature_vector)[0][1]
                predictions[label] = round(float(prob), 3)
            except Exception as exc:
                print(f"[predict] Error predicting '{label}': {exc}")
                predictions[label] = 0.0
    else:
        # ── Mock prediction pipeline ─────────────────────────────────────
        predictions = {
            label: _deterministic_prob(smiles, label)
            for label in label_cols
        }

    return {
        "molecular_weight": round(raw["mol_weight"], 3),
        "logp":             round(raw["logp"], 3),
        "qed":              round(raw["qed"], 3),
        "predictions":      predictions,
    }
