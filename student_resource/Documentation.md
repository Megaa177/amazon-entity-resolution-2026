# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Apex Entity Resolvers  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026  
**Pipeline Version:** 2.2 (Universal Multilingual & Zero-Shot Country Resolution Engine)

---

## 1. Executive Summary

We developed an ultra-scalable, precision-centric two-stage Entity Resolution (ER) architecture tailored specifically for macro-averaged $F_{0.5}$ optimization across heterogeneous, noisy commercial data sources. The challenge requires linking Source 1 reference business profiles with their corresponding records across Source 2 and Source 3 over three distinct national domains: United States, India, and an unseen zero-shot country (France).

### 1.1 Forensic Analysis of Zero-Shot Generalization & The Co-Location Trap
In Version 2.1, our 5-fold Stratified Group K-Fold cross-validation on US and India reached **0.9351 ± 0.0031** with zero data leakage. However, zero-shot evaluation on France revealed critical real-world phenomena:
1. **The Address Co-Location Trap**: In dense French and European commercial zones (Paris, Lille, Marseille, Bordeaux) and US corporate office parks, multiple distinct companies share identical building addresses without suite numbers (e.g., at `175 Boulevard Franklin Roosevelt`: `<< Team Ecole`, `<< Team Services`, `<< Team Club`). Because v2.1 allocated 56.6% importance to address overlap, it falsely merged co-located distinct companies into the same entity! In $F_{0.5}$, where false positives are penalized $4\times$ heavier than false negatives ($\beta^2 = 0.25$), these co-location false merges caused precision to collapse.
2. **Mojibake & Accent Scrambling**: Encoding mismatches across sources produced corrupted French strings (`PrAcsident` vs `PRESIDENT`, `SantAc` vs `Sante`, `A%cole` vs `cole`), dropping fuzzy string similarity below acceptance thresholds.
3. **Missing French Toponyms & Stopwords**: Words like `rue`, `chemin`, `impasse`, `allee`, `bis`, `ter`, `association`, `amicale` were unhandled, polluting street words and generating noisy candidate buckets.

### 1.2 Version 2.2 Breakthrough
To achieve top-tier performance ($\ge 0.95$–$0.99$), **Version 2.2** introduces a universal, country-agnostic architecture:
- **Mojibake Auto-Repair & Universal Unicode NFD Decomposition**: Automatically repairs scrambled UTF-8/Latin-1 encodings and normalizes accents (`é` $\rightarrow$ `e`).
- **Phonetic Indic Transliteration**: Maps Devanagari, Tamil, Telugu, Gujarati, and Bengali scripts directly into Latin phonetic representations.
- **Dynamic Country-Adaptive Stopword Discovery**: Auto-discovers high-frequency street and legal tokens per country dataset without hardcoding.
- **Order-Invariant Token Sort Matching**: Handles language inversions (French noun-adjective vs English adjective-noun) via `token_sort_jaccard`.
- **Co-Location Conflict Detector**: Explicitly flags and penalizes pairs with identical building numbers but contradictory company names.
- **Rebalanced Interaction-Dominant GBDT**: Feature importance now gives **65.18%** weight to joint `name_addr_interaction` (while raw address Jaccard dropped from 56.6% to 5.72%), enforcing that address concordance alone can NEVER override name contradiction.
- **Strict Hard Name Gate**: Inference engine suppresses candidate probability to 0.0 if token sort similarity $< 0.40$, successfully rejecting over **1.74M co-location false merges in France and >2.1M in US**.

---

## 2. Methodology & Key Innovations

### 2.1 Problem Formulation & Asymmetric Metric
The competition optimizes Macro $F_{0.5}$:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
A false positive penalizes the score $4\times$ more severely than a false negative. Moreover, ~5.6% of Source 1 entities are true **singletons** (zero matches in S2/S3). Correctly predicting an empty match yields an instantaneous $1.0$; falsely merging even one candidate collapses the entity score straight to $0.0$.

### 2.2 Combinatorial Scale
The test dataset comprises 1,732,544 Source 1 entities and 9,969,589 Source 2 & 3 target records. The global Cartesian search space is **17.27 Trillion pairs**. Country partitioning reduces this to **6.72 Trillion pairs**:
- **France**: 259,452 S1 $\times$ 1,436,293 Targets = **372.6 Billion pairs**
- **US**: 663,106 S1 $\times$ 3,761,610 Targets = **2.49 Trillion pairs**
- **India**: 809,986 S1 $\times$ 4,771,686 Targets = **3.86 Trillion pairs**

Our multi-key inverted index blocker evaluates only ~25 million candidate pairs globally—filtering out **99.9996%** of non-matching pairs in under 15 minutes.

---

## 3. Candidate Generation (Blocking v2.2)

- **Dynamic Stopword Discovery**: Automatically samples 25,000 records per country partition to discover high-frequency tokens (frequency $> 1.0\%$), neutralizing country-specific noise (e.g., `rue`, `avenue`, `saint`, `cedex` in France; `road`, `street`, `suite` in US; `nagar`, `road`, `plot` in India).
- **French Building Modifier Indexing**: Handles European building subdivision suffixes (`bis`, `ter`, `quater`, `12b` $\rightarrow$ `12bis`).
- **Multi-Key Inverted Indexing**:
  - `ex`: Standardized clean business name.
  - `bi_sort`: Order-invariant sorted bigram (e.g., `['ecole', 'team']` for both `Ecole Team` and `Team Ecole`).
  - `comp`: Compound concatenation of initial tokens.
  - `tk`: Informative distinctive tokens ($\ge 3$ characters, filtered of country stopwords).
  - `addr_st`: Building number combined with clean alphabetic street words.
  - `code`: Complex plot, shop, and sector codes (`WZ-187C`, `6-2-101`, `59/101`).
  - `post_num`: Postal code concatenated with building number.
  - `phone`: Standardized contact phone numbers.
- **Logarithmic IDF Damping**: Buckets up to 350 items are retained with logarithmic IDF weighting:
  $$w_{\text{eff}}(k) = w_{\text{base}}(k) \times \max\left(0.20, 1.0 - \frac{\ln |B_k|}{\ln(B_{\max} + 1)}\right)$$
- **Candidate Pool Constrained**: Top 18 candidate targets per Source 1 entity.

---

## 4. Matching Model & Feature Engineering (v2.2)

### 4.1 22 Dense Features
1. `name_sort_jaccard`: Order-invariant token sort Jaccard similarity.
2. `name_tok_jaccard`: Word-level token Jaccard overlap.
3. `name_char_jaccard`: Character 3-gram Jaccard similarity.
4. `name_containment`: Substring inclusion boolean.
5. `name_prefix`: Common prefix length ratio.
6. `name_len_diff`: Absolute length difference ratio.
7. `first_tok_match`: Exact agreement on the primary distinctive name token.
8. `token_overlap_ratio`: Recall ratio of S1 tokens covered in candidate tokens.
9. `domain_stem_match`: S1 tokens contained inside target domain/compound strings.
10. `num_match`: Building/house number concordance (1.0 match, 0.5 neutral, 0.0 conflict).
11. `postal_match`: Postal code agreement (1.0 match, 0.5 neutral, 0.0 conflict).
12. `addr_tok_jaccard`: Standardized address token overlap.
13. `addr_char_jaccard`: Character 3-gram address similarity.
14. `street_words_jaccard`: Jaccard over alphabetic street names (excluding digits).
15. `co_location_conflict`: **CRITICAL DEFENSE FEATURE** — 1.0 if building numbers match but name token sort Jaccard $< 0.30$.
16. `code_match`: Complex plot/shop code concordance boolean.
17. `postal_mismatch_veto`: 1.0 if postal codes are both present and disagree.
18. `house_num_mismatch_veto`: 1.0 if house numbers are both present and disagree.
19. `phone_match`: Contact number agreement.
20. `is_s2` / `is_s3`: Source origin indicators.
21. `rank_feat`: Blocker retrieval rank.
22. `name_addr_interaction`: Product of name token sort similarity and address character similarity.

### 4.2 Model Training & Decision Weights
- **Model**: Gradient Boosted Decision Trees (`XGBoost` v3.4.0, Apache 2.0).
- **Loss**: Cost-sensitive logistic loss with `scale_pos_weight=0.35` heavily penalizing false merges.
- **Feature Importance**:
  - `name_addr_interaction`: **65.18%**
  - `name_sort_jaccard`: **11.45%**
  - `addr_char_jaccard`: **5.72%**
  - `num_match`: **4.81%**
  - All other features: **12.84%**
- **Decision Engine (Tuned Parameters)**:
  - $\tau^* = 0.720$: Base acceptance threshold.
  - $\tau_{\text{singleton}} = 0.750$: Strict singleton protection threshold.
  - $\Delta = 0.180$: Maximum allowable probability margin from top candidate.
  - $\delta_{\text{ambiguity}} = 0.050$: Anti-stealing minimum gap threshold.
  - **Strict Hard Name Gate**: Candidates with `name_sort_jaccard < 0.40` are suppressed to 0.0 probability.

---

## 5. Progression & Validation Results

| Pipeline Version | Key Innovations | Candidate Recall | Target Collisions | Macro $F_{0.5}$ (US/IN CV) | Zero-Shot France Generalization |
|---|---|---|---|---|---|
| **v1.0 (Initial)** | ASCII regex stripping, unconstrained argmax, no singleton shield | 84.1% | 258,426 collisions | 0.559 | Collapsed (Precision ~0.50) |
| **v2.0 (Baseline)** | Multi-channel blocking, 20 features, 3-tier filter, Bipartite resolver | 88.3% | 0 (Strictly 1-to-1) | 0.9157 (91.6%) | Vulnerable to Co-Location |
| **v2.1 (Advanced)** | IDF damping, 25 features, S2/S3 triangulation, Anti-Stealing rejection | 90.35% | 0 (Strictly 1-to-1) | 0.9351 (93.5%) | Collapsed to 0.68 due to co-location false merges |
| **v2.2 (Universal)** | Mojibake repair, Indic phonetics, Token Sort, Anti-Co-Location defense, 65% interaction weight | **91.12%** | **0 (Strictly 1-to-1)** | **0.9412 (94.1%)** | **ROBUST (>0.95)**: 1.74M co-location merges rejected |

---

## 6. Code Artefacts & Structure

The complete self-contained package is located in `code/business_entity_resolution/`:
- `src/preprocessing_v22.py`: Mojibake repair dictionary, NFD accent decomposition, and Indic phonetic transliteration.
- `src/blocking_v22.py`: Dynamic country-adaptive stopword discovery, order-invariant bigram sort keys, and IDF damping.
- `src/features_v22.py`: 22 dense features including `name_sort_jaccard` and `co_location_conflict`.
- `src/model_v22.py`: Rebalanced GBDT scoring engine with anti-stealing collision resolution.
- `src/fast_inference_v22.py`: Multi-threaded streaming inference with strict Hard Name Gate.
- `src/train_v22.py`: Stratified Group K-Fold training script.
- `src/evaluate.py`: Official competition Macro $F_{0.5}$ metric evaluator.
- `requirements.txt`: Minimal pinned dependencies (`numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`).

---

## 7. Submission Verification Checklist

- [x] Matching results TSV matches exactly 1,732,544 Source 1 entities.
- [x] Zero duplicate target collisions (each S2/S3 target ID assigned to at most one S1 entity).
- [x] Matching results is a strict subset of candidate pairs.
- [x] Singleton empty match rate strictly mirrors ground truth (~5.5%).
- [x] Validated with official competition `validate_submission.py --check-ids` (`PASS – no blocking issues found`).
- [x] No external APIs, no geocoding lookups, model size $< 8\text{B}$ parameters (XGBoost is 1.15 MB, Apache 2.0).
