"""
DrugMirror – SHAP Explainability
==================================
╔══════════════════════════════════════════════════════════════════════════╗
║  MOCK MODE NOTICE                                                        ║
║                                                                          ║
║  This file generates fake but DETERMINISTIC SHAP feature importances     ║
║  for frontend development while the ML model is being trained.           ║
║                                                                          ║
║  Once drugmirror_models.pkl, label_cols.pkl, and feature_cols.pkl are   ║
║  placed in backend/models/, this will automatically switch to real       ║
║  SHAP values on next restart — NO code changes needed.                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import hashlib

# ---------------------------------------------------------------------------
# Feature name → human-readable label mapping (for real-mode SHAP output)
# ---------------------------------------------------------------------------
FEATURE_LABEL_MAP: dict[str, str] = {
    "logp":                 "Lipophilicity (LogP)",
    "tpsa":                 "Polar Surface Area (TPSA)",
    "mol_weight":           "Molecular Weight",
    "qed":                  "Drug-likeness (QED)",
    "logBB_est":            "Blood-Brain Barrier Estimate",
    "cns_mpo_score":        "CNS Drug-likeness Score",
    "frac_sp3":             "3D Character (sp3 fraction)",
    "basic_nitrogen_count": "Basic Nitrogen Count",
    "hbd":                  "Hydrogen Bond Donors",
    "hba":                  "Hydrogen Bond Acceptors",
    "rotatable_bonds":      "Rotatable Bonds",
    "aromatic_rings":       "Aromatic Rings",
    "heavy_atoms":          "Heavy Atom Count",
    "affinity_mean":        "Binding Affinity Estimate",
}

# Human-readable labels used in mock mode (matches UI design requirements)
MOCK_FEATURE_LABELS: list[str] = [
    "Lipophilicity (LogP)",
    "Polar Surface Area (TPSA)",
    "Molecular Weight",
    "Drug-likeness (QED)",
    "Blood-Brain Barrier Estimate",
    "CNS Drug-likeness Score",
    "3D Character (sp3 fraction)",
    "Aromatic Ring Pattern #1",
    "Aromatic Ring Pattern #2",
    "Hydrogen Bond Donor Pattern",
]


def _deterministic_shap(smiles: str, feature: str) -> float:
    """Return a deterministic SHAP-like value in [-0.3, 0.3].

    Uses SHA-256 of the concatenated smiles+feature so the same inputs always
    produce the same output — useful for stable frontend mock data.

    Parameters
    ----------
    smiles : str
        Canonical SMILES string.
    feature : str
        Human-readable feature label used as part of the hash seed.

    Returns
    -------
    float
        A value in [-0.3, 0.3] rounded to 4 decimal places.
    """
    seed = f"{smiles.strip()}::shap::{feature}"
    digest = hashlib.sha256(seed.encode()).hexdigest()
    int_val = int(digest[:8], 16)
    # Map [0, 0xFFFFFFFF] → [-0.3, 0.3]
    normalised = int_val / 0xFFFFFFFF  # [0, 1]
    raw = (normalised - 0.5) * 0.6    # [-0.3, 0.3]
    return round(raw, 4)


def get_shap_features(smiles: str, top_n: int = 10) -> list[dict]:
    """Return the top-N SHAP feature importances for the given molecule.

    Operates in one of two modes:

    * **MOCK** (``predict.MODEL_LOADED = False``): returns a hardcoded list of
      10 dicts with realistic human-readable feature names and deterministic
      SHAP values derived from the SMILES hash.

    * **REAL** (``predict.MODEL_LOADED = True``): computes genuine SHAP values
      via ``shap.TreeExplainer`` on the first model in ``drugmirror_models.pkl``
      and returns the ``top_n`` features sorted by absolute SHAP value.

    Parameters
    ----------
    smiles : str
        A valid SMILES string.
    top_n : int
        Number of features to return (default 10).

    Returns
    -------
    list[dict]
        Each element::

            {
                "feature": str,   # human-readable feature name
                "label":   str,   # side-effect category (mock: "overall")
                "value":   float  # SHAP value (negative = protective)
            }
    """
    import predict  # late import to avoid circular dependency

    if not predict.MODEL_LOADED:
        # ── Mock SHAP values ─────────────────────────────────────────────
        return [
            {
                "feature": feat_label,
                "label":   "overall",  # mock doesn't target a specific class
                "value":   _deterministic_shap(smiles, feat_label),
            }
            for feat_label in MOCK_FEATURE_LABELS[:top_n]
        ]

    # ── Real SHAP values ─────────────────────────────────────────────────
    try:
        import shap
        import numpy as np
        from features import extract_features, assemble_feature_vector

        # Use the first trained model and its label for SHAP
        first_label = predict.label_cols[0]
        clf = predict.models[first_label]

        extracted = extract_features(smiles)
        feature_vector = assemble_feature_vector(extracted, predict.feature_cols)

        explainer  = shap.TreeExplainer(clf)
        shap_vals  = explainer.shap_values(feature_vector)

        # For binary classifiers shap_values returns a list [neg_class, pos_class]
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class
        shap_vals = np.array(shap_vals).flatten()

        # Build column name list (fp_0..fp_2047 then named descriptors)
        if predict.feature_cols is not None:
            all_cols = predict.feature_cols
        else:
            fp_cols   = [f"fp_{i}" for i in range(2048)]
            desc_cols = list(extracted["descriptors"].keys())
            cns_cols  = list(extracted["cns_features"].keys())
            all_cols  = fp_cols + desc_cols + cns_cols

        # Sort by absolute SHAP value, pick top_n
        sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:top_n]

        results = []
        for idx in sorted_idx:
            raw_name = all_cols[idx] if idx < len(all_cols) else f"fp_{idx}"
            # Resolve human-readable label
            if raw_name in FEATURE_LABEL_MAP:
                readable = FEATURE_LABEL_MAP[raw_name]
            elif raw_name.startswith("fp_"):
                bit_num  = raw_name.split("_")[1]
                readable = f"Structural Pattern #{bit_num}"
            else:
                readable = raw_name

            results.append({
                "feature": readable,
                "label":   first_label,
                "value":   round(float(shap_vals[idx]), 4),
            })

        return results

    except Exception as exc:
        print(f"[shap_explain] Real SHAP computation failed: {exc}. Falling back to mock.")
        return [
            {
                "feature": feat_label,
                "label":   "overall",
                "value":   _deterministic_shap(smiles, feat_label),
            }
            for feat_label in MOCK_FEATURE_LABELS[:top_n]
        ]
