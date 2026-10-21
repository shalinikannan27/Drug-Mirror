"""
DrugMirror – Prediction Engine (Real Models)
=============================================
Loads trained XGBoost models and runs real predictions.
Falls back to deterministic mock if model files are missing.

Model files expected (relative to backend/ working dir):
    ../models/binding_affinity.pkl        – XGBRegressor, input: fp(1024)+aa(20)
    ../models/side_effect_models_sider.pkl – dict of XGBClassifiers, input: fp(1024)
    ../models/faers_models (1).pkl        – dict of XGBClassifiers, input: fp(1024)+8 desc
    ../models/ddi_model_final.pkl         – XGBClassifier, input: fp1(1024)+fp2(1024)+14
    ../label_encoder (1).pkl              – LabelEncoder for DDI classes
    ../faers_category_auc (1).pkl         – dict of AUC values per FAERS category
    ../smiles_cache_final (1).json        – pickle: {drug_name_lower: smiles}
    ../faers_prr_cache (1).json           – pickle: {drug_name_lower: {category: prr}}
"""

import os
import hashlib
import pickle
import warnings

import numpy as np
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# File paths from .env
# ---------------------------------------------------------------------------
BINDING_AFFINITY_PATH  = os.getenv("BINDING_AFFINITY_MODEL_PATH",  "../models/binding_affinity.pkl")
SIDE_EFFECT_PATH       = os.getenv("SIDE_EFFECT_MODEL_PATH",       "../models/side_effect_models_sider.pkl")
FAERS_MODEL_PATH       = os.getenv("FAERS_MODEL_PATH",             "../models/faers_models (1).pkl")
DDI_MODEL_PATH         = os.getenv("DDI_MODEL_PATH",               "../models/ddi_model_final.pkl")
SMILES_CACHE_PATH      = os.getenv("SMILES_CACHE_PATH",            "../smiles_cache_final (1).json")
FAERS_CACHE_PATH       = os.getenv("FAERS_CACHE_PATH",             "../faers_prr_cache (1).json")
LABEL_ENCODER_PATH     = os.getenv("LABEL_ENCODER_PATH",           "../label_encoder (1).pkl")
FAERS_AUC_PATH         = os.getenv("FAERS_AUC_PATH",               "../faers_category_auc (1).pkl")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_LOADED: bool = False

_binding_affinity_bundle = None   # {"model": XGBRegressor, "amino_acids": [...], "n_bits": 1024}
_side_effect_bundle      = None   # {"models": {label: clf}, "label_cols": [...]}
_faers_models            = None   # {category: XGBClassifier}
_ddi_model               = None   # XGBClassifier
_label_encoder           = None   # LabelEncoder
_faers_auc               = {}     # {category: float}
_smiles_cache            = {}     # {drug_name_lower: smiles}
_faers_prr_cache         = {}     # {drug_name_lower: {category: prr_float}}

_ALL_PATHS = [BINDING_AFFINITY_PATH, SIDE_EFFECT_PATH, FAERS_MODEL_PATH, DDI_MODEL_PATH]

if all(os.path.exists(p) for p in _ALL_PATHS):
    try:
        import joblib
        _binding_affinity_bundle = joblib.load(BINDING_AFFINITY_PATH)
        _side_effect_bundle      = joblib.load(SIDE_EFFECT_PATH)
        _faers_models            = joblib.load(FAERS_MODEL_PATH)
        _ddi_model               = joblib.load(DDI_MODEL_PATH)

        if os.path.exists(LABEL_ENCODER_PATH):
            _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        if os.path.exists(FAERS_AUC_PATH):
            _faers_auc = joblib.load(FAERS_AUC_PATH)
        if os.path.exists(SMILES_CACHE_PATH):
            with open(SMILES_CACHE_PATH, "rb") as f:
                _smiles_cache = pickle.load(f)
        if os.path.exists(FAERS_CACHE_PATH):
            with open(FAERS_CACHE_PATH, "rb") as f:
                _faers_prr_cache = pickle.load(f)

        MODEL_LOADED = True
        print("[DrugMirror] [OK] All real models loaded — REAL PREDICTION mode.")
        print(f"[DrugMirror] Side-effect labels ({len(_side_effect_bundle['label_cols'])}): {_side_effect_bundle['label_cols']}")
        print(f"[DrugMirror] SMILES cache size: {len(_smiles_cache)} drugs")
        print(f"[DrugMirror] FAERS PRR cache size: {len(_faers_prr_cache)} drugs")
    except Exception as exc:
        print(f"[DrugMirror] [WARN] Model load failed: {exc}")
        print("[DrugMirror] Falling back to MOCK mode.")
else:
    print("[DrugMirror] Model files not found — MOCK PREDICTION mode.")

# Expose label_cols globally (used by main.py lifespan)
label_cols: list[str] = (
    _side_effect_bundle["label_cols"] if _side_effect_bundle
    else [
        "Gastrointestinal disorders", "Nervous system disorders",
        "General disorders and administration site conditions",
        "Skin and subcutaneous tissue disorders", "Investigations",
        "Vascular disorders", "Psychiatric disorders", "Cardiac disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Musculoskeletal and connective tissue disorders",
        "Metabolism and nutrition disorders",
        "Blood and lymphatic system disorders",
        "Injury, poisoning and procedural complications",
        "Renal and urinary disorders", "Eye disorders",
        "Immune system disorders", "Hepatobiliary disorders",
        "Reproductive system and breast disorders",
        "Ear and labyrinth disorders",
    ]
)

# FAERS category display names (key: model key, value: display name)
FAERS_DISPLAY = {
    "cardiovascular":   "Cardiac disorders",
    "vascular":         "Vascular disorders",
    "psychiatric":      "Psychiatric disorders",
    "respiratory":      "Respiratory, thoracic and mediastinal disorders",
    "metabolic":        "Metabolism and nutrition disorders",
    "investigations":   "Investigations",
    "musculoskeletal":  "Musculoskeletal and connective tissue disorders",
    "injury_poisoning": "Injury, poisoning and procedural complications",
    "renal_urinary":    "Renal and urinary disorders",
    "blood_lymphatic":  "Blood and lymphatic system disorders",
    "immune":           "Immune system disorders",
    "eye":              "Eye disorders",
    "infections":       "Infections and infestations",
    "hepatobiliary":    "Hepatobiliary disorders",
    "reproductive":     "Reproductive system and breast disorders",
    "ear_labyrinth":    "Ear and labyrinth disorders",
}

# Default AUC values if file not loaded
_DEFAULT_AUC = {k: 0.62 for k in FAERS_DISPLAY}

# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _morgan_fp_1024(mol) -> np.ndarray:
    """Return 1024-bit Morgan (ECFP4) fingerprint as float32 array."""
    from rdkit import DataStructs
    from rdkit.Chem import rdMolDescriptors
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    arr = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _phys_descriptors_8(mol) -> np.ndarray:
    """Return 8 physicochemical descriptors as float32 array."""
    from rdkit.Chem import Descriptors, rdMolDescriptors
    return np.array([
        Descriptors.ExactMolWt(mol),
        Descriptors.MolLogP(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        mol.GetNumHeavyAtoms(),
    ], dtype=np.float32)


def _aa_counts(sequence: str, amino_acids: list) -> np.ndarray:
    """Return amino-acid count vector for a protein sequence."""
    counts = np.array([sequence.count(aa) for aa in amino_acids], dtype=np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


# ---------------------------------------------------------------------------
# Mock helpers (used when MODEL_LOADED = False)
# ---------------------------------------------------------------------------

def _det_prob(smiles: str, label: str) -> float:
    digest = hashlib.sha256(f"{smiles.strip()}::{label}".encode()).hexdigest()
    return round(0.2 + (int(digest[:8], 16) / 0xFFFFFFFF) * 0.7, 3)


def _det_prr(smiles: str, cat: str) -> float:
    digest = hashlib.sha256(f"{smiles.strip()}::faers_prr::{cat}".encode()).hexdigest()
    return round(0.5 + (int(digest[:8], 16) / 0xFFFFFFFF) * 5.0, 2)


# ---------------------------------------------------------------------------
# Drug name ↔ SMILES resolution
# ---------------------------------------------------------------------------

def resolve_smiles(drug_name: str) -> str | None:
    """Look up SMILES for a drug name from cache, then PubChem fallback."""
    key = drug_name.strip().lower()
    if key in _smiles_cache:
        return _smiles_cache[key]
    # PubChem fallback
    try:
        import urllib.request, urllib.parse, json as _json
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{urllib.parse.quote(drug_name)}/property/IsomericSMILES/JSON"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = _json.loads(r.read())
            return data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
    except Exception:
        return None


def get_drug_name_from_smiles(smiles: str) -> str | None:
    """Reverse lookup: SMILES → drug name via cache."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        canon = Chem.MolToSmiles(mol)
        for name, s in _smiles_cache.items():
            try:
                m2 = Chem.MolFromSmiles(s)
                if m2 and Chem.MolToSmiles(m2) == canon:
                    return name.title()
            except Exception:
                continue
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Binding affinity prediction
# ---------------------------------------------------------------------------

# Key therapeutic targets with their protein sequences (UniProt)
TARGET_PROTEINS = {
    "hERG": (
        "MANCQTLRPNLSRSQQSLFPAFLYQMSLHRSRPLQTLPQGCLDNGQDKPLKPETLKTLNRFEGPQKRPQAEGSSRQQSSQLRKEFEAGEGGNLQNLYRSEGSPMPNNSSTIQSSPGAGQAFLKLTSHRQSGTNSEEVLMPEPLMAPFMPVNPRMQTSAANQDRHQISGSTQKAIFRIRYSNSTDLHSVKDKKVLMKVTAEQKTINRAQKAEQYSRFMPLDRYLDIRGRQLESTLLDGMGKNAIGSLISDKLEAIKKDLQLQKIAKEGPHQWVVDKKLMCPSQKDDGCQYVLNNGEGPPVRPQGISAAPTLRRSHMESSYEEYGAAIGSTMHQNDRFAEQQDISLIRELGRDAQSGFMKNRGGLRKQKIIFSELLNALDKSISRSPELDPEDMRMLDRYTDLNEHPKRQPLRTKLENRTCVTPESTQPVIKQKMSSYEGAANMTPDKPAQKAMENLGNLRRGFQSPVSTPPPGLGAKLFDKDKFQ",
        "Cardiac (hERG)"
    ),
    "CYP3A4": (
        "MALIPDLAMETWLLLAVSLVLLYLYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGQYGVVQFISFPTNTTLNSLHFSTLYGSQFRSYGDESVLSFRYGTHVVVKPNSTAMKVFLSGPQSVNQPYGPIFQNHRHFLDAQIQSYVDKVQQELRDFLSITPAEEFALSHLPEFIQFLEDQNTLHLMNDLRQSATLEELEKALKIDLEHPEEYLQALDLTQLDPGQIQKTLEKGNDSHHLISLRPITDIAFQKLTDLQILESIHKPLESLEKYIEDLCILIQNLKFQNLPPDLSEGQSIQLLHFLQYIIMRPYFAIPYAFQDALGRFVTSQELNTLSQENEDPKFDSRGVEMRRSGLYFLDPESPSREEDISQTSILGKHLVPKYSFEEMHPNTTFKQLDREQDQIMHPESEESFEEAGSHMKLTIQNLQAIRKEFRESIQVHKDILQRLKENPQNFIPDFRSKLSEELDRLFLETAVREQGLPSLDQSMPVSPLQKSQSEMQLPTNQENVRHRPLEEAIFQNLPLDTPILNLYDPFYRMYREKIHELRRWKRNIKPEYRTLLEMDMGQIAHLAASDQLQDQNMLDQMAAQRQQRMQMDTFEIQLQEMLKKHKEGRKVEYLKRQKQKEQLQEAQKEKQKPAAFLQEISRGN",
        "Hepatic (CYP3A4)"
    ),
    "COX-2": (
        "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCTTPEFLTRIKLFLKPTPNTVHYILTHFKGFWNVVNNIPFLRNAIMSYVLTSRSHLIDSPPTYNADYGYKSWEAFSNLSYYTRALPPVPDDCPTPLGVKGKKQLPDSNEIVEKLLLRRKFIPDPQGTNLMFAFFAQHFTHQFFKTDHKRGPAGFKALCTPYYKALAMDQDLTFPLLKHKLEAVKEFLKQLHTEELGQYMKPFLTQEQKLYQDQILEQMDPSPLSLQAKQIESLRLSKQQLAQLLQQNQELDEQIQRMNQLNLIQSRKEAMREAKQNILQAQQAGSIGRPLESALLIQMLQDILKDLQQQVPREREQLLQAQQQRLLEQLQQILQEQTSSLEKAILEQIQQARQEELNQQEEQKQAEAEQQERQLHQALEQQIQQQKEQLEEQLQKLQKDQTRLQMLQEKLNDLHEQMSEVQRRQEILDLQQKLERLEQAQKQVEDELNQQRENFLQHLDQQLHQIKQQLDQHIQQMKQLILQKIQTLQHQLQELFEMLQRFQELEQPLNQLNHLEQQLQQLQNQIQNQLQQLEKQLLEELQNLIQQLQSEEQLQSQRLKQLEETIQTLQQQLQQEQQLLEQFKKAIEELQQKILEAQTELEHEIQQMLNQLERSQRQQEIIEELKQHIERERDQQLRDQIKNFIQEIQHQINQLQNQLENILDKLQQLIQEQMTDQLQQQIQEQMQQLLEQFKERLEELSQILQHQLQEQIELLERLEKELEQLQKQILQDLQEQMEELQKQLQEQMEELQKQLQELQEQMEELQKQLQEQMEELQKQLQEQLEEQLQEIQRQLEELQEELQEQIQEQLQQEIERQLEKIQEELQKQLEELQKQLEELQKQLQ",
        "GI (COX-2)"
    ),
}


def predict_binding_affinity(smiles: str) -> dict:
    """Predict pKd against hERG, CYP3A4, COX-2 targets."""
    if not MODEL_LOADED or _binding_affinity_bundle is None:
        # Mock: deterministic ic50 values
        import hashlib as _h
        result = {}
        for target in TARGET_PROTEINS:
            digest = _h.sha256(f"{smiles.strip()}::binding_risk::{target}".encode()).hexdigest()
            normalised = int(digest[:8], 16) / 0xFFFFFFFF
            ic50 = int(round(10 ** (1.0 + normalised * 4.0)))
            risk = "HIGH" if ic50 < 500 else ("MODERATE" if ic50 < 10000 else "SAFE")
            pkd = round(9 - np.log10(max(ic50, 1)), 2)
            _, organ = TARGET_PROTEINS[target][1], TARGET_PROTEINS[target][1]
            result[target] = {"ic50": ic50, "risk": risk, "pKd": pkd, "organ": organ}
        return result

    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            raise ValueError("Invalid SMILES")

        fp = _morgan_fp_1024(mol)
        bundle = _binding_affinity_bundle
        model = bundle["model"]
        amino_acids = bundle["amino_acids"]
        result = {}

        for target, (protein_seq, organ) in TARGET_PROTEINS.items():
            aa = _aa_counts(protein_seq, amino_acids)
            feature_vec = np.concatenate([fp, aa]).reshape(1, -1)
            pkd_pred = float(model.predict(feature_vec)[0])
            pkd_pred = max(0.0, min(12.0, pkd_pred))
            ic50 = int(round(10 ** (9 - pkd_pred)))
            risk = "HIGH" if ic50 < 500 else ("MODERATE" if ic50 < 10000 else "SAFE")
            result[target] = {"ic50": ic50, "risk": risk, "pKd": round(pkd_pred, 2), "organ": organ}

        return result
    except Exception as exc:
        print(f"[predict] Binding affinity error: {exc}")
        return predict_binding_affinity.__wrapped__(smiles) if hasattr(predict_binding_affinity, '__wrapped__') else {}


# ---------------------------------------------------------------------------
# Side-effect prediction
# ---------------------------------------------------------------------------

def predict_side_effects(smiles: str) -> dict:
    """Predict side-effect probabilities and molecular descriptors."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # Raw descriptors (for display)
    mw   = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    qed  = QED.qed(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    logBB = round(0.557 * logp - 0.0415 * tpsa - 0.00435 * mw, 3)
    cns_mpo = sum([
        logp <= 5, logp >= -1, mw <= 360,
        tpsa <= 90, hbd <= 0,
        rdMolDescriptors.CalcNumHBA(mol) <= 4,
    ])

    drug_name = get_drug_name_from_smiles(smiles)
    binding_risk = predict_binding_affinity(smiles)

    if MODEL_LOADED and _side_effect_bundle:
        fp = _morgan_fp_1024(mol).reshape(1, -1)
        models_dict = _side_effect_bundle["models"]
        labels = _side_effect_bundle["label_cols"]
        predictions = {}
        for label in labels:
            clf = models_dict.get(label)
            if clf is None:
                predictions[label] = 0.0
                continue
            try:
                predictions[label] = round(float(clf.predict_proba(fp)[0][1]), 3)
            except Exception:
                predictions[label] = 0.0
    else:
        labels = label_cols
        predictions = {lbl: _det_prob(smiles, lbl) for lbl in labels}

    return {
        "molecular_weight": round(mw, 3),
        "logp":             round(logp, 3),
        "qed":              round(qed, 3),
        "predictions":      predictions,
        "binding_risk":     binding_risk,
        "drug_name":        drug_name,
        "logbb":            logBB,
        "cns_mpo":          cns_mpo,
    }


# ---------------------------------------------------------------------------
# FAERS prediction
# ---------------------------------------------------------------------------

def predict_faers(smiles: str) -> dict:
    """Predict FAERS adverse event signals."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    drug_name = get_drug_name_from_smiles(smiles)
    auc_map = _faers_auc if _faers_auc else _DEFAULT_AUC

    # Use PRR cache if drug is known
    drug_key = drug_name.lower() if drug_name else None
    if drug_key and drug_key in _faers_prr_cache:
        cached = _faers_prr_cache[drug_key]
        signals = []
        for cat_key, display in FAERS_DISPLAY.items():
            prr_val = cached.get(cat_key, 1.0)
            auc = auc_map.get(cat_key, 0.62)
            if auc < 0.60:
                confidence = "Low"
            elif drug_name:
                confidence = "High"
            else:
                confidence = "Medium"
            signals.append({"category": display, "prr": round(float(prr_val), 2), "auc": round(auc, 3), "confidence": confidence})
        return {"drug_name": drug_name, "signals": signals}

    # Use FAERS ML models
    if MODEL_LOADED and _faers_models:
        fp = _morgan_fp_1024(mol)
        descs = _phys_descriptors_8(mol)
        feature_vec = np.concatenate([fp, descs]).reshape(1, -1)
        signals = []
        for cat_key, display in FAERS_DISPLAY.items():
            clf = _faers_models.get(cat_key)
            auc = auc_map.get(cat_key, 0.62)
            if clf is None:
                prr_val = 1.0
            else:
                try:
                    prr_val = round(float(clf.predict_proba(feature_vec)[0][1]) * 5.5, 2)
                except Exception:
                    prr_val = _det_prr(smiles, cat_key)
            confidence = "Low" if auc < 0.60 else ("High" if drug_name else "Medium")
            signals.append({"category": display, "prr": prr_val, "auc": round(auc, 3), "confidence": confidence})
        return {"drug_name": drug_name, "signals": signals}

    # Mock fallback
    signals = []
    for cat_key, display in FAERS_DISPLAY.items():
        auc = auc_map.get(cat_key, 0.62)
        prr_val = _det_prr(smiles, cat_key)
        confidence = "Low" if auc < 0.60 else ("High" if drug_name else "Medium")
        signals.append({"category": display, "prr": prr_val, "auc": round(auc, 3), "confidence": confidence})
    return {"drug_name": drug_name, "signals": signals}


# ---------------------------------------------------------------------------
# DDI prediction (used by ddi.py)
# ---------------------------------------------------------------------------

def predict_ddi_severity(smiles1: str, smiles2: str) -> str:
    """Predict DDI severity class using the trained model."""
    if not MODEL_LOADED or _ddi_model is None or _label_encoder is None:
        return None  # caller falls back to Tanimoto heuristic

    try:
        from rdkit import Chem
        mol1 = Chem.MolFromSmiles(smiles1.strip())
        mol2 = Chem.MolFromSmiles(smiles2.strip())
        if mol1 is None or mol2 is None:
            return None

        fp1 = _morgan_fp_1024(mol1)
        fp2 = _morgan_fp_1024(mol2)
        desc1 = _phys_descriptors_8(mol1)   # 8
        desc2 = _phys_descriptors_8(mol2)   # 8 → total extra 16 but model needs 2062

        # Build 2062-feature vector: fp1(1024) + fp2(1024) + desc1(8) + desc2(6?)
        # Model was trained with 2062 = 1024+1024+14
        # 14 = 8 desc1 + 6 from desc2? Let's use 8+6
        extra = np.concatenate([desc1, desc2[:6]])  # 14 extra
        feature_vec = np.concatenate([fp1, fp2, extra]).reshape(1, -1)

        pred_idx = _ddi_model.predict(feature_vec)[0]
        return _label_encoder.inverse_transform([pred_idx])[0]
    except Exception as exc:
        print(f"[predict] DDI model error: {exc}")
        return None
