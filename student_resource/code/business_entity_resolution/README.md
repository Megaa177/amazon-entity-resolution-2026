# Business Entity Resolution Solution

This package contains the self-contained, end-to-end entity resolution pipeline designed to resolve noisy business entities across three heterogeneous data sources (Source 1, Source 2, and Source 3) under a precision-heavy macro $F_{0.5}$ metric.

---

## 1. System Overview & Architecture

The repository contains two production pipelines:

### Version 2.0 (High-Precision Macro $F_{0.5}$ Baseline)
- **Candidate Recall**: 88.3%+ with multi-channel inverted indexing (`src/blocking.py`).
- **Feature Set**: 20 dense similarity features (`src/features.py`).
- **Model**: Cost-sensitive XGBoost (`src/model_v2.joblib`) with 3-tier threshold filtering ($\tau^* = 0.70$, $\tau_{\text{single}} = 0.75$, $\Delta = 0.15$).
- **Conflict Resolution**: Global bipartite mutual exclusivity ensuring each target record matches at most one Source 1 entity (0 duplicate collisions).
- **Validation Macro $F_{0.5}$**: **0.9157 (91.6%)**.

### Version 2.1 (Advanced Multi-Country Resolution Engine)
- **Candidate Recall**: **90.35%** recall with precomputed logarithmic IDF damping across buckets up to 350 items (`src/blocking_v21.py`).
- **Feature Set**: 25 dense signals including pure street words Jaccard, character 2-gram Dice, token coverage, and first-token matching (`src/features_v21.py`).
- **Model**: Cost-sensitive GBDT (`src/model_v21.joblib`) with $\tau^* = 0.720$, $\tau_{\text{single}} = 0.740$, $\Delta = 0.180$.
- **S2 $\leftrightarrow$ S3 Sibling Triangulation**: Automatically promotes true sibling pairs sharing house numbers and distinctive tokens that fall slightly below single-candidate thresholds.
- **Anti-Stealing Ambiguity Rejection**: Rejects contested target candidates where competing claims are within $\delta < 0.05$, aggressively shielding precision against catastrophic $F_{0.5}$ penalties.
- **Validation Macro $F_{0.5}$**: **0.9344 (93.44%)**.

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── preprocessing.py         # Multilingual text cleaning, suffix stripping, address parsing
│   ├── blocking.py              # Version 2.0 Multi-channel candidate blocker
│   ├── blocking_v21.py          # Version 2.1 IDF-damped candidate blocker
│   ├── features.py              # Version 2.0 20-feature extractor
│   ├── features_v21.py          # Version 2.1 25-feature extractor
│   ├── model.py                 # Version 2.0 GBDT classifier & threshold tuner
│   ├── model_v21.py             # Version 2.1 GBDT, sibling triangulation & anti-stealing
│   ├── model_v2.joblib          # Version 2.0 trained model checkpoint
│   ├── model_v21.joblib         # Version 2.1 trained model checkpoint
│   ├── fast_inference.py        # Version 2.0 multi-country streaming inference
│   ├── fast_inference_v21.py    # Version 2.1 multi-country streaming inference
│   └── evaluate.py              # Competition Macro F0.5 evaluation metric implementation
├── README.md                    # Reproduction guide
└── requirements.txt             # Pinned dependencies
```

---

## 3. Setup & Installation

Install dependencies from the package root:
```bash
pip install -r requirements.txt
```

---

## 4. How to Reproduce

### Run Version 2.1 Inference (Recommended, Validation F0.5: 0.9344)
```bash
python src/fast_inference_v21.py \
    --output_match ../../output/matching_results_v21.tsv \
    --output_cand ../../output/candidate_pairs_v21.tsv
```

### Run Version 2.0 Inference (Baseline, Validation F0.5: 0.9157)
```bash
python src/fast_inference.py
```

### Official Submission Format Validation
Run the official competition validator:
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results_v21.tsv \
    --candidate ../../output/candidate_pairs_v21.tsv \
    --test-dir ../../dataset/test
```
