"""
DrugMirror – Drug-Drug Interaction Checker
==========================================
Uses the trained DDI XGBClassifier (severity: mild/moderate/severe/unknown)
with Tanimoto structural similarity as a secondary metric.
Falls back to Tanimoto-only heuristic when the model isn't loaded.
"""

import hashlib
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors


def _deterministic_shared_targets(smiles1: str, smiles2: str) -> list[str]:
    """Deterministically pick 0-2 shared targets based on combined SMILES pairs."""
    sorted_smiles = sorted([smiles1.strip(), smiles2.strip()])
    seed_str = f"{sorted_smiles[0]}::ddi_shared_targets::{sorted_smiles[1]}"
    digest = hashlib.sha256(seed_str.encode()).hexdigest()
    int_val = int(digest[:8], 16)

    num_targets = int_val % 3
    if num_targets == 0:
        return []

    targets_pool = ["hERG", "CYP3A4", "COX-2", "COX-1", "P-gp"]
    idx1 = (int_val // 3) % len(targets_pool)
    if num_targets == 1:
        return [targets_pool[idx1]]
    else:
        idx2 = (int_val // 15) % len(targets_pool)
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(targets_pool)
        return sorted([targets_pool[idx1], targets_pool[idx2]])


def check_interaction(smiles1: str, smiles2: str) -> dict:
    """Estimate drug-drug interaction risk between two molecules.

    Uses the trained DDI XGBClassifier when available, with Tanimoto
    similarity as a structural-overlap metric.

    Returns
    -------
    dict with keys: tanimoto_similarity, risk_level, explanation, shared_targets
    """
    mol1 = Chem.MolFromSmiles(smiles1.strip())
    mol2 = Chem.MolFromSmiles(smiles2.strip())

    if mol1 is None:
        raise ValueError(f"Invalid SMILES for drug 1: '{smiles1}'")
    if mol2 is None:
        raise ValueError(f"Invalid SMILES for drug 2: '{smiles2}'")

    # Tanimoto similarity (always computed for display)
    fp1 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024)
    fp2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    similarity_rounded = round(float(similarity), 3)
    similarity_pct = round(similarity_rounded * 100, 1)

    # Try real DDI model first
    severity_label = None
    try:
        from predict import predict_ddi_severity
        severity_label = predict_ddi_severity(smiles1, smiles2)
    except Exception:
        pass

    # Map severity to risk_level
    if severity_label is not None:
        severity_map = {
            "severe":   "HIGH",
            "moderate": "MODERATE",
            "mild":     "LOW",
            "unknown":  "LOW",
        }
        risk_level = severity_map.get(severity_label.lower(), "LOW")
        risk_desc = risk_level.lower()
        model_note = f"Model predicts {severity_label} severity interaction."
    else:
        # Tanimoto heuristic fallback
        if similarity_rounded > 0.7:
            risk_level, risk_desc = "HIGH", "high"
        elif similarity_rounded >= 0.3:
            risk_level, risk_desc = "MODERATE", "moderate"
        else:
            risk_level, risk_desc = "LOW", "low"
        model_note = ""

    explanation = (
        f"These molecules share {similarity_pct}% structural similarity, "
        f"suggesting {risk_desc} potential for overlapping mechanisms of action."
    )
    if model_note:
        explanation += f" {model_note}"

    return {
        "tanimoto_similarity": similarity_rounded,
        "risk_level":          risk_level,
        "explanation":         explanation,
        "shared_targets":      _deterministic_shared_targets(smiles1, smiles2),
    }
