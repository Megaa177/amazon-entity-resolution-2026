# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Apex Entity Resolvers  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026  
**Pipeline Version:** 2.5 (Full Haystack Hard Negative Trained Universal Engine)

---

## 1. Executive Summary

We developed an ultra-scalable, precision-centric two-stage Entity Resolution (ER) architecture engineered specifically for macro-averaged $F_{0.5}$ optimization across massive, noisy commercial datasets. The challenge links Source 1 reference business profiles with their corresponding records across Source 2 and Source 3 over three distinct national domains: United States, India, and an unseen zero-shot country (France).

### 1.1 Forensic Analysis of Leaderboard Collapses (0.68 & 0.50 Root Cause Analysis)
Through systematic forensic auditing, we uncovered the two root causes of previous leaderboard drops:
1. **The Toy Training Distribution Discrepancy (The 0.68 Collapse)**:
   - Early versions (v2.0–v2.2) filtered training targets via `if parts[0] in needed_targets:`, loading only 69,144 targets (0.67% of the 10.3M database).
   - In that isolated 0.67% pool, true targets were isolated with **zero realistic distractors**, producing an artificially inflated cross-validation score of ~0.98.
   - When deployed against the real 10-million target test haystack, the model encountered dense office buildings, commercial parks, and lookalike brand names that it had never seen during training, causing false positive merges that dragged precision down and collapsed the score to 0.68.
2. **The Precision Sensitivity of $F_{0.5}$ (The 0.50 Collapse)**:
   - The competition metric is $F_{0.5}$:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   - Because false positives are penalized $4\times$ heavier than false negatives, precision is paramount.
   - When v2.4 relaxed the manual gating heuristics in India to increase recall, the uncalibrated model merged lookalikes (e.g., individual owners with commercial trusts sharing a street name), collapsing precision from 65% to 47% and driving $F_{0.5}$ down to 0.50.

### 1.2 Version 2.5 Breakthrough: Full Haystack Training & Hard Negative Mining
To permanently solve these failure modes, **Version 2.5** introduces:
- **Full Haystack Training Engine (`train_full_haystack.py`)**: Scanned all **10,320,219 targets** in `train_source2.tsv` and `train_source3.tsv`. Indexed 1,282,692 targets matching active query keys, generating 328,642 training pairs: 69,144 true positives vs **259,498 REAL hard negative distractors** (ratio: 3.75 hard negatives per positive).
- **Honest 5-Fold Stratified Group K-Fold CV ($0.9019 \pm 0.0015$)**: Evaluated against the real unconstrained 10M haystack with zero data leakage:
  - Fold 1: 0.9047 | Fold 2: 0.9016 | Fold 3: 0.9017 | Fold 4: 0.9011 | Fold 5: 0.9003
  - **Mean Honest CV: $90.19\%$** (Full fit: $90.78\%$).
- **Cost-Sensitive Asymmetric Optimization (`scale_pos_weight=0.20`)**: The GBDT heavily penalizes false merges during tree splitting, learning the exact mathematical decision boundary between true entities and lookalikes.
- **Calibrated Decision Thresholds**:
  $$\tau^* = 0.680, \quad \tau_{\text{singleton}}^* = 0.740, \quad \Delta^* = 0.170, \quad \delta_{\text{ambiguity}}^* = 0.050$$
- **Entity Distinctiveness-Adaptive Thresholding**:
  $$\tau_{\text{eff}} = \tau + (0.5 - \text{distinctiveness}) \times 0.08$$
  $$\tau_{\text{singleton, eff}} = \tau_{\text{singleton}} + (0.5 - \text{dist}) \times 0.06$$
  Generic entities ("Apex Corp") face stricter thresholds, while distinctive entities ("Pecoraro Forensic Engineering") capture legitimate variations.
- **Anti-Stealing Bipartite Mutual Exclusivity**: Ensures every target is assigned to at most one Source 1 entity, strictly eliminating multi-tenant duplicate collisions.

---

## 2. Methodology & Key Innovations

### 2.1 Problem Formulation & Asymmetric Metric
The competition optimizes Macro $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
A false positive penalizes the score $4\times$ more severely than a false negative. Moreover, ~5.5% of Source 1 entities are true **singletons** (zero matches in S2/S3). Correctly predicting an empty match yields an instantaneous $1.0$; falsely merging even one candidate collapses the entity score straight to $0.0$.

### 2.2 Combinatorial Scale
The test dataset comprises 1,732,544 Source 1 entities and 9,969,589 Source 2 & 3 target records. The global Cartesian search space is **17.27 Trillion pairs**. Country partitioning reduces this to **6.72 Trillion pairs**:
- **France**: 259,452 S1 $\times$ 1,434,993 Targets = **372.3 Billion pairs**
- **US**: 663,106 S1 $\times$ 3,817,031 Targets = **2.53 Trillion pairs**
- **India**: 809,986 S1 $\times$ 4,717,565 Targets = **3.82 Trillion pairs**

Our multi-key inverted index blocker evaluates only ~10 million candidate pairs globally—filtering out **99.9998%** of non-matching pairs in under 10 minutes.

---

## 3. Candidate Generation (Blocking v2.2)

- **Dynamic Stopword Discovery**: Automatically samples 30,000 records per country partition to discover high-frequency tokens (frequency $> 3.0\%$), neutralizing country-specific noise (e.g., `rue`, `avenue`, `saint`, `cedex` in France; `road`, `street`, `suite` in US; `nagar`, `road`, `plot` in India).
- **French Building Modifier Indexing**: Handles European building subdivision suffixes (`bis`, `ter`, `quater`, `12b` $\rightarrow$ `12bis`).
- **Multi-Key Inverted Indexing**:
  - `ex`: Standardized clean business name.
  - `bi_sort`: Order-invariant sorted bigram (e.g., `['ecole', 'team']` for both `Ecole Team` and `Team Ecole`).
  - `tk`: Informative distinctive tokens ($\ge 3$ characters, filtered of country stopwords).
  - `addr_st`: Building number combined with clean alphabetic street words.
  - `code`: Complex plot, shop, and sector codes (`WZ-187C`, `6-2-101`, `59/101`).
  - `post_num`: Postal code concatenated with building number.
  - `phone`: Standardized contact phone numbers.
- **Logarithmic IDF Damping**: Buckets up to 350 items are retained with logarithmic IDF weighting:
  $$w_{\text{eff}}(k) = w_{\text{base}}(k) \times \max\left(0.20, 1.0 - \frac{\ln |B_k|}{\ln(B_{\max} + 1)}\right)$$
- **Candidate Pool Constrained**: Top 18 candidate targets per Source 1 entity.

---

## 4. Matching Model & Feature Engineering (v2.3/v2.5)

### 4.1 25 Dense Features
1. `name_exact`: Exact clean name match.
2. `name_sort_jaccard`: Order-invariant token sort Jaccard similarity.
3. `name_tok_jaccard`: Word-level token Jaccard overlap.
4. `name_dice_2g`: Character 2-gram Dice coefficient.
5. `name_char_jaccard`: Character 3-gram Jaccard similarity.
6. `token_overlap_ratio`: Recall ratio of S1 tokens covered in candidate tokens.
7. `name_len_diff`: Absolute length difference ratio.
8. `num_match`: Building/house number concordance (1.0 match, 0.5 neutral, 0.0 conflict).
9. `postal_match`: Postal code agreement (1.0 match, 0.5 neutral, 0.0 conflict).
10. `addr_tok_jaccard`: Standardized address token overlap.
11. `addr_char_jaccard`: Character 3-gram address similarity.
12. `street_words_jaccard`: Jaccard over alphabetic street names (excluding digits).
13. `co_location_conflict`: Flags pairs with matching numbers but contradictory names.
14. `house_num_mismatch_veto`: 1.0 if house numbers are both present and disagree.
15. `postal_mismatch_veto`: 1.0 if postal codes are both present and disagree.
16. `first_tok_match`: Exact agreement on the primary distinctive name token.
17. `phone_match`: Contact number agreement.
18. `code_match`: Complex plot/shop code concordance boolean.
19. `is_s2` / `is_s3`: Source origin indicators.
20. `rank_feat`: Blocker retrieval rank.
21. `name_addr_interaction`: Product of name Dice similarity and address character similarity.
22. `acronym_match`: Detects initialisms (e.g., `IBM` $\leftrightarrow$ `International Business Machines`).
23. `lcp_ratio`: Longest Common Prefix ratio for truncation detection.
24. `distinctiveness_score`: Score $[0.0, 1.0]$ based on name token count and specificity.

### 4.2 Model Training & Calibration
- **Model**: Gradient Boosted Decision Trees (`XGBoost` v3.4.0, Apache 2.0).
- **Hard Negative Training**: Trained against 259,498 real negative distractors mined across 10.3M targets.
- **Loss**: Cost-sensitive logistic loss with `scale_pos_weight=0.20`.
- **Decision Engine (Calibrated Parameters)**:
  - $\tau^* = 0.680$: Base acceptance threshold.
  - $\tau_{\text{singleton}}^* = 0.740$: Strict singleton protection threshold.
  - $\Delta^* = 0.170$: Maximum allowable probability margin from top candidate.
  - $\delta_{\text{ambiguity}}^* = 0.050$: Anti-stealing minimum gap threshold.

---

## 5. Progression & Validation Results

| Pipeline Version | Key Innovations | Hard Negatives Mined | Target Collisions | Macro $F_{0.5}$ (Honest CV) | Leaderboard Result |
|---|---|---|---|---|---|
| **v2.0** | Multi-channel blocking, 20 features, Bipartite resolver | 0 (Toy subset) | 0 | 0.9157 (Toy CV) | Overfitted to toy data |
| **v2.1** | IDF damping, S2/S3 triangulation, Anti-Stealing | 0 (Toy subset) | 0 | 0.9351 (Toy CV) | Collapsed to 0.68 on real haystack |
| **v2.2** | Mojibake repair, Indic phonetics, Co-Location defense | 0 (Toy subset) | 0 | 0.9412 (Toy CV) | 0.683 (Public Leaderboard) |
| **v2.4** | Staged FR/US from v2.2, relaxed Indian gate | 0 (Toy subset) | 0 | 0.9200 (Toy CV) | 0.500 (False merges in India) |
| **v2.5** | **Full Haystack Training (10.3M targets), 259k Real Distractors, Distinctiveness Thresholding** | **259,498 Distractors** | **0 (Strictly 1-to-1)** | **0.9019 ± 0.0015 (REAL CV)** | **Full Fit: 0.9078** |

---

## 6. Code Artefacts & Structure

The complete self-contained package is located in `code/business_entity_resolution/`:
- `src/preprocessing_v23.py`: Mojibake repair dictionary, NFD accent decomposition, Indic phonetic transliteration, acronym extraction, and distinctiveness scoring.
- `src/blocking_v22.py`: Dynamic country-adaptive stopword discovery, order-invariant bigram sort keys, and IDF damping.
- `src/features_v23.py`: 25 dense features including `name_addr_interaction`, `acronym_match`, `lcp_ratio`, and `co_location_conflict`.
- `src/model_v23.py`: Cost-sensitive GBDT scoring engine with sibling triangulation and anti-stealing collision resolution.
- `src/fast_inference_v25.py`: Universal multi-country streaming inference engine with distinctiveness-adaptive thresholding.
- `src/train_full_haystack.py`: Full haystack training script scanning 10.3M targets and mining 259k real distractors.
- `src/model_full_haystack.joblib`: Serialized model checkpoint trained on the real 10M haystack.
- `src/evaluate.py`: Official competition Macro $F_{0.5}$ metric evaluator.
- `requirements.txt`: Minimal pinned dependencies (`numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`).

---

## 7. Submission Verification Checklist

- [x] Matching results TSV matches exactly 1,732,544 Source 1 entities.
- [x] Zero duplicate target collisions (each S2/S3 target ID assigned to at most one S1 entity).
- [x] Matching results is a strict subset of candidate pairs.
- [x] Singleton empty match rate strictly mirrors ground truth (~5.5%).
- [x] Validated with official competition `validate_submission.py --check-ids` (`PASS – no blocking issues found`).
- [x] No external APIs, no geocoding lookups, model size $< 8\text{B}$ parameters (XGBoost is 1.14 MB, Apache 2.0).
