"""
DrugMirror – Drug–Drug Interaction (DDI) Checker
==================================================
Computes Tanimoto similarity between two molecules' Morgan fingerprints and
classifies the interaction risk level.  This module is **fully functional**
now — it has no dependency on the trained ML model files.
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def check_interaction(smiles1: str, smiles2: str) -> dict:
    """Estimate the drug–drug interaction risk between two molecules.

    Uses Tanimoto similarity on 2048-bit Morgan (ECFP4) fingerprints as a
    structural-overlap proxy.  While this is a simplified heuristic, it
    gives the frontend a meaningful risk signal without requiring a trained
    DDI model.

    Risk thresholds
    ---------------
    * ``> 0.7``   → HIGH
    * ``0.3–0.7`` → MODERATE
    * ``< 0.3``   → LOW

    Parameters
    ----------
    smiles1 : str
        SMILES string for the first drug.
    smiles2 : str
        SMILES string for the second drug.

    Returns
    -------
    dict
        ``{``
        ``"tanimoto_similarity": float,   # rounded to 3 dp``
        ``"risk_level": str,             # "HIGH" | "MODERATE" | "LOW"``
        ``"explanation": str``           # human-readable one-liner``
        ``}``

    Raises
    ------
    ValueError
        If either SMILES string is invalid.
    """
    mol1 = Chem.MolFromSmiles(smiles1.strip())
    mol2 = Chem.MolFromSmiles(smiles2.strip())

    if mol1 is None:
        raise ValueError(f"Invalid SMILES for drug 1: '{smiles1}'")
    if mol2 is None:
        raise ValueError(f"Invalid SMILES for drug 2: '{smiles2}'")

    # Generate Morgan fingerprints (ECFP4 equivalent)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)

    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    similarity_rounded = round(float(similarity), 3)
    similarity_pct = round(similarity_rounded * 100, 1)

    # Classify risk
    if similarity_rounded > 0.7:
        risk_level = "HIGH"
        risk_desc  = "high"
    elif similarity_rounded >= 0.3:
        risk_level = "MODERATE"
        risk_desc  = "moderate"
    else:
        risk_level = "LOW"
        risk_desc  = "low"

    explanation = (
        f"These molecules share {similarity_pct}% structural similarity, "
        f"suggesting {risk_desc} potential for overlapping mechanisms of action."
    )

    return {
        "tanimoto_similarity": similarity_rounded,
        "risk_level":          risk_level,
        "explanation":         explanation,
    }
