"""
DrugMirror – Feature Extraction Pipeline
=========================================
Computes all molecular features required by the DrugMirror ML model from a
raw SMILES string.  Designed to run end-to-end **without** the trained model
files so the pipeline can be tested immediately.

Sections
--------
1. Constants / scaling parameters  (TODO: fill in from training statistics)
2. validate_smiles()               – SMILES sanity check
3. extract_features()              – returns raw fingerprint + descriptor dicts
4. assemble_feature_vector()       – concatenates everything in model order

Integration note
----------------
Once *feature_cols.pkl* is available, update assemble_feature_vector() so it
reads the exact column ordering used during training.  Everything else stays
the same.
"""

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import (
    Descriptors,
    rdMolDescriptors,
    QED,
    AllChem,
)

# ---------------------------------------------------------------------------
# 1. CONSTANTS / SCALING PARAMETERS
# ---------------------------------------------------------------------------
# TODO: Replace these placeholder means/stds with the values your teammate
#       exports after training (e.g. from scaler.mean_ and scaler.scale_).
#       Current values (mean=0, std=1) implement a no-op scaling so the full
#       pipeline runs end-to-end right now.

DESCRIPTOR_SCALING = {
    # "descriptor_name": (mean, std)
    "mol_weight":         (0.0, 1.0),  # TODO: fill in from training stats
    "logp":               (0.0, 1.0),  # TODO: fill in from training stats
    "hbd":                (0.0, 1.0),  # TODO: fill in from training stats
    "hba":                (0.0, 1.0),  # TODO: fill in from training stats
    "tpsa":               (0.0, 1.0),  # TODO: fill in from training stats
    "rotatable_bonds":    (0.0, 1.0),  # TODO: fill in from training stats
    "aromatic_rings":     (0.0, 1.0),  # TODO: fill in from training stats
    "heavy_atoms":        (0.0, 1.0),  # TODO: fill in from training stats
    "qed":                (0.0, 1.0),  # TODO: fill in from training stats
    "affinity_mean":      (0.0, 1.0),  # TODO: fill in from training stats
    "logBB_est":          (0.0, 1.0),  # TODO: fill in from training stats
    "cns_mpo_score":      (0.0, 1.0),  # TODO: fill in from training stats
    "frac_sp3":           (0.0, 1.0),  # TODO: fill in from training stats
    "basic_nitrogen_count": (0.0, 1.0),  # TODO: fill in from training stats
}

# Fingerprint settings – must match values used during training
FP_RADIUS = 2
FP_N_BITS = 2048

# Affinity placeholder – training set median from BindingDB
# TODO: Replace with live BindingDB lookup once the API integration is built
AFFINITY_MEAN_PLACEHOLDER = 2.4203  # training-set median pChEMBL value


# ---------------------------------------------------------------------------
# 2. SMILES VALIDATION
# ---------------------------------------------------------------------------

def validate_smiles(smiles: str) -> bool:
    """Return True if *smiles* is a valid, parseable SMILES string.

    Uses RDKit's ``MolFromSmiles`` which returns ``None`` for invalid input.

    Parameters
    ----------
    smiles : str
        Raw SMILES string to validate.

    Returns
    -------
    bool
        ``True`` if the molecule can be parsed, ``False`` otherwise.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3. CORE FEATURE EXTRACTION
# ---------------------------------------------------------------------------

def _scale(value: float, name: str) -> float:
    """Apply z-score scaling using the constants table.

    With the current no-op constants (mean=0, std=1) this is an identity
    function.  Replace the table entries to activate real scaling.
    """
    mean, std = DESCRIPTOR_SCALING.get(name, (0.0, 1.0))
    if std == 0.0:
        return 0.0
    return (value - mean) / std


def extract_features(smiles: str) -> dict:
    """Extract all molecular features from a SMILES string.

    The return value intentionally keeps fingerprint, physicochemical
    descriptors, and CNS features **separate** because the correct
    concatenation order is determined by *feature_cols.pkl* (not yet
    available).  See ``assemble_feature_vector()`` for the fallback order.

    Parameters
    ----------
    smiles : str
        A valid SMILES string (canonicalised internally).

    Returns
    -------
    dict with keys:
        ``fingerprint``  – np.ndarray of shape (2048,), dtype float32
        ``descriptors``  – dict of 10 physicochemical features (scaled)
        ``cns_features`` – dict of 4 CNS penetration features (scaled)

    Raises
    ------
    ValueError
        If the SMILES string cannot be parsed by RDKit.
    """
    # --- Canonicalise ---------------------------------------------------------
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'")
    canonical_smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(canonical_smiles)  # re-parse canonical form

    # --- Morgan fingerprint ---------------------------------------------------
    fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_N_BITS)
    fp_array = np.zeros((FP_N_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_gen, fp_array)

    # --- Physicochemical descriptors ------------------------------------------
    mol_weight      = Descriptors.ExactMolWt(mol)
    logp            = Descriptors.MolLogP(mol)
    hbd             = rdMolDescriptors.CalcNumHBD(mol)
    hba             = rdMolDescriptors.CalcNumHBA(mol)
    tpsa            = rdMolDescriptors.CalcTPSA(mol)
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
    aromatic_rings  = rdMolDescriptors.CalcNumAromaticRings(mol)
    heavy_atoms     = mol.GetNumHeavyAtoms()
    qed_val         = QED.qed(mol)
    # affinity_mean: hardcoded training-set median (no BindingDB lookup yet)
    affinity_mean   = AFFINITY_MEAN_PLACEHOLDER  # TODO: replace with live lookup

    descriptors = {
        "mol_weight":      _scale(mol_weight,      "mol_weight"),
        "logp":            _scale(logp,             "logp"),
        "hbd":             _scale(hbd,              "hbd"),
        "hba":             _scale(hba,              "hba"),
        "tpsa":            _scale(tpsa,             "tpsa"),
        "rotatable_bonds": _scale(rotatable_bonds,  "rotatable_bonds"),
        "aromatic_rings":  _scale(aromatic_rings,   "aromatic_rings"),
        "heavy_atoms":     _scale(heavy_atoms,      "heavy_atoms"),
        "qed":             _scale(qed_val,          "qed"),
        "affinity_mean":   _scale(affinity_mean,    "affinity_mean"),
    }

    # --- CNS penetration features --------------------------------------------
    # logBB estimate: Abraham & Acree (2003) regression
    logBB_est = 0.557 * logp - 0.0415 * tpsa - 0.00435 * mol_weight

    # CNS MPO score: Wager 2010 (6-rule, 0–6 scale)
    cns_mpo_score = sum([
        1 if logp <= 5   else 0,
        1 if logp >= -1  else 0,
        1 if mol_weight <= 360 else 0,
        1 if tpsa <= 90  else 0,
        1 if hbd <= 0    else 0,
        1 if hba <= 4    else 0,
    ])

    # Fraction of sp3 carbons (3D character proxy)
    frac_sp3 = rdMolDescriptors.CalcFractionCSP3(mol)

    # Basic nitrogen count: N atoms with ≥1 attached H
    basic_nitrogen_count = sum(
        1 for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() >= 1
    )

    cns_features = {
        "logBB_est":           _scale(logBB_est,            "logBB_est"),
        "cns_mpo_score":       _scale(cns_mpo_score,        "cns_mpo_score"),
        "frac_sp3":            _scale(frac_sp3,             "frac_sp3"),
        "basic_nitrogen_count": _scale(basic_nitrogen_count, "basic_nitrogen_count"),
    }

    # Store raw (unscaled) values for display purposes
    _raw = {
        "mol_weight": mol_weight,
        "logp":       logp,
        "qed":        qed_val,
        "tpsa":       tpsa,
        "hbd":        hbd,
        "hba":        hba,
    }

    return {
        "fingerprint":  fp_array,
        "descriptors":  descriptors,
        "cns_features": cns_features,
        "_raw":         _raw,         # unscaled values for API response display
    }


# ---------------------------------------------------------------------------
# 4. FEATURE VECTOR ASSEMBLY
# ---------------------------------------------------------------------------

def assemble_feature_vector(extracted: dict, feature_cols: list | None = None) -> np.ndarray:
    """Concatenate all features into a single 1-D numpy array for the model.

    TODO: Once *feature_cols.pkl* is available this function should re-order
          the named descriptor/CNS columns to match the exact training column
          ordering.  Right now it uses a fixed fallback order:

              [fp_0..fp_2047] ++ [10 descriptors] ++ [4 CNS features]

    Parameters
    ----------
    extracted : dict
        Output from :func:`extract_features`.
    feature_cols : list | None
        Ordered list of column names from *feature_cols.pkl*.  When ``None``
        the fallback fixed order is used.

    Returns
    -------
    np.ndarray
        Shape (2062,) float32 vector ready to pass to the sklearn models.
    """
    fp = extracted["fingerprint"]  # shape (2048,)

    # Fallback fixed ordering (matches training only if teammate used same order)
    # TODO: Replace with feature_cols-driven ordering once pkl is available
    DESC_ORDER = [
        "mol_weight", "logp", "hbd", "hba", "tpsa",
        "rotatable_bonds", "aromatic_rings", "heavy_atoms", "qed", "affinity_mean",
    ]
    CNS_ORDER = ["logBB_est", "cns_mpo_score", "frac_sp3", "basic_nitrogen_count"]

    if feature_cols is not None:
        # TODO: implement proper column-driven ordering
        # For now: pull named features in feature_cols order, skip fp_ columns
        # which are already prepended above.
        named = {**extracted["descriptors"], **extracted["cns_features"]}
        named_vec = np.array(
            [named[c] for c in feature_cols if c in named], dtype=np.float32
        )
    else:
        desc_vec = np.array(
            [extracted["descriptors"][k] for k in DESC_ORDER], dtype=np.float32
        )
        cns_vec = np.array(
            [extracted["cns_features"][k] for k in CNS_ORDER], dtype=np.float32
        )
        named_vec = np.concatenate([desc_vec, cns_vec])

    return np.concatenate([fp, named_vec]).reshape(1, -1)
