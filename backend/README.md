# DrugMirror Backend — Developer Handoff

## Quick Start

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Visit **http://localhost:8000/docs** for the auto-generated Swagger UI.

---

## File Map

```
backend/
├── main.py           — FastAPI app, CORS, all 4 endpoints
├── features.py       — SMILES → fingerprint + descriptors + CNS features
├── predict.py        — Side-effect prediction (mock/real auto-switch)
├── shap_explain.py   — SHAP feature importances (mock/real auto-switch)
├── molecule.py       — 2D molecule depiction → base64 PNG
├── ddi.py            — Drug–drug interaction via Tanimoto similarity
├── models/
│   └── .gitkeep      — Drop the 3 pkl files here to go live
├── .env              — Model paths and CORS origins
├── .gitignore        — Excludes *.pkl, .venv, .env
└── requirements.txt
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status + mode |
| `POST` | `/predict` | Side-effect probabilities + SHAP |
| `GET` | `/molecule?smiles=...` | Base64 2D depiction |
| `POST` | `/ddi` | Drug–drug interaction check |

### Sample requests

```bash
# Health
curl http://localhost:8000/health

# Predict (Aspirin)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'

# Molecule image
curl "http://localhost:8000/molecule?smiles=CC(=O)Oc1ccccc1C(=O)O"

# DDI check
curl -X POST http://localhost:8000/ddi \
  -H "Content-Type: application/json" \
  -d '{"smiles1": "CC(=O)Oc1ccccc1C(=O)O", "smiles2": "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"}'
```

---

## Going Live (switching from MOCK → REAL)

Once your teammate finishes training, they need to export **3 files**:

### From the training notebook:

```python
import joblib

# 1. Model dict  — one sklearn estimator per label
joblib.dump(models_dict, "drugmirror_models.pkl")
# models_dict = {"gi_issues": clf1, "skin_reactions": clf2, ...}

# 2. Label list  — exactly matching the dict keys, in order
joblib.dump(label_cols_list, "label_cols.pkl")

# 3. Feature column order  — critical for correct vector assembly
joblib.dump(feature_cols_list, "feature_cols.pkl")
# feature_cols_list = ["fp_0", "fp_1", ..., "fp_2047", "mol_weight", "logp", ...]
```

### Drop them in:

```
backend/models/drugmirror_models.pkl
backend/models/label_cols.pkl
backend/models/feature_cols.pkl
```

**Restart the server** — it auto-detects the files and switches to REAL mode. No code changes needed.

---

## TODOs for teammate integration

| File | Line(s) | What to do |
|------|---------|------------|
| `features.py` | `DESCRIPTOR_SCALING` dict | Fill in `mean` and `std` per feature from `scaler.mean_` / `scaler.scale_` |
| `features.py` | `AFFINITY_MEAN_PLACEHOLDER` | Replace with live BindingDB pChEMBL lookup when available |
| `features.py` | `assemble_feature_vector()` | Verify fallback column order matches `feature_cols.pkl`; extend `fp_`-column handling if `feature_cols` includes fingerprint bits by name |
| `predict.py` | model file paths | `.env` already points to correct paths; confirm your teammate uses the same filenames |

---

## CNS Features Reference

| Feature | Formula / Method | Source |
|---------|-----------------|--------|
| `logBB_est` | `0.557·logP − 0.0415·TPSA − 0.00435·MW` | Abraham & Acree 2003 |
| `cns_mpo_score` | 6-rule Wager score (0–6) | Wager 2010 |
| `frac_sp3` | `CalcFractionCSP3` | RDKit |
| `basic_nitrogen_count` | N atoms with ≥1 H attached | RDKit atom iteration |
