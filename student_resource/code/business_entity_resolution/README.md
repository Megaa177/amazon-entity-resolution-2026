# Business Entity Resolution Solution

This package contains the self-contained, end-to-end entity resolution pipeline designed to resolve noisy business entities across three heterogeneous data sources (Source 1, Source 2, and Source 3) under a precision-heavy macro $F_{0.5}$ metric.

---

## 1. System Overview & Architecture

The solution uses a high-performance two-stage entity resolution architecture:
1. **Dynamic Country-Partitioned Multi-Index Candidate Generation (Blocking)**:
   - Partitions search space by country (`US`, `India`, `France`).
   - Multi-key indexing using exact standardized name, token bigrams, rare distinctive tokens, address building numbers, and postal codes.
   - Frequency pruning to discard generic tokens and keep candidate sets tight ($\le 15$ per S1 entity).
2. **Dense Feature Engineering**:
   - Computes 14 discriminative lexical, token-overlap, address-component, prefix, and source-interaction signals for each pair.
3. **Calibrated Gradient Boosted Trees (XGBoost)**:
   - Evaluates pairwise candidate matches.
   - Optimizes probability decision threshold $\tau^*$ strictly against the competition's macro-averaged $F_{0.5}$ formula.
   - Built-in singleton preservation layer that outputs an empty match set when candidate confidence falls below $\tau^*$, protecting the critical $1.0$ score on singletons.

---

## 2. Directory Structure

```
business_entity_resolution/
├── src/
│   ├── preprocessing.py    # Multilingual text cleaning, suffix stripping, address parsing
│   ├── blocking.py         # Multi-channel candidate blocker & inverted index
│   ├── features.py         # Pairwise similarity feature extractor
│   ├── model.py            # XGBoost classifier & macro F0.5 threshold optimizer
│   ├── evaluate.py         # Competition Macro F0.5 evaluation metric implementation
│   ├── pipeline.py         # End-to-end training and inference pipeline
│   └── model.joblib        # Pre-trained optimized model checkpoint
├── README.md               # Reproduction guide
└── requirements.txt        # Pinned dependencies
```

---

## 3. Setup & Installation

Install dependencies from the package root:
```bash
pip install -r requirements.txt
```

---

## 4. How to Reproduce

### Step 1: Model Training & Threshold Optimization
To train the XGBoost classifier from ground truth data and optimize the $F_{0.5}$ threshold:
```bash
python src/pipeline.py --mode train --train-dir ../../dataset/train --sample-gt 15000
```
This fits the model, performs grouped validation, finds the optimal threshold $\tau^* = 0.725$ (achieving ~0.895 validation Macro $F_{0.5}$), and saves `model.joblib`.

### Step 2: Test Inference & File Generation
To run candidate generation and model scoring over the test dataset:
```bash
python src/pipeline.py --mode predict --test-dir ../../dataset/test --output-dir ../../output
```
This generates the two required output files:
- `output/matching_results.tsv` (Leaderboard submission file)
- `output/candidate_pairs.tsv` (Blocking candidate pairs)

### Step 3: Submission Format Validation
Run the official challenge validator from `student_resource/`:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
