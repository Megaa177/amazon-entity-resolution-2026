# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Apex Entity Resolvers  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026  
**Pipeline Version:** 2.1 (Advanced Multi-Country Resolution Engine)

---

## 1. Executive Summary

We developed an ultra-scalable, precision-centric two-stage Entity Resolution (ER) architecture tailored specifically for macro-averaged $F_{0.5}$ optimization across heterogeneous, noisy commercial data sources. The challenge requires linking Source 1 reference business profiles with their corresponding records across Source 2 and Source 3 over three distinct national domains (United States, India, and an unseen zero-shot country, France).

In Version 2.0, we introduced multi-channel blocking, 20 dense features, cost-sensitive GBDT scoring, and Global Mutual Exclusivity, which elevated the validation score to **0.9157 (91.6%)** and eradicated 258,426 duplicate target collisions.

In **Version 2.1**, we extended the architecture to achieve **0.9344 (93.44%)** validation Macro $F_{0.5}$ through four major innovations:
1. **Logarithmic IDF Frequency Damping**: Instead of hard-deleting candidate buckets larger than 200 items (which inadvertently pruned 13.8% of valid long-tail matches in Version 2.0), Version 2.1 indexes buckets up to 350 items and applies precomputed logarithmic IDF weight attenuation:
   $$w_{\text{eff}}(k) = w_{\text{base}}(k) \times \max\left(0.20, 1.0 - \frac{\ln |B_k|}{\ln(B_{\max} + 1)}\right)$$
   This elevated candidate blocker recall from **84.1% to 90.35%** with zero combinatorial blowup.
2. **Dense 25-Feature Space**: Added pure street-words Jaccard (excluding noise digits), character 2-gram Dice coefficient, first-token exact anchor, phone concordance, and token recall coverage.
3. **Cross-Source Sibling Triangulation ($S2 \leftrightarrow S3$)**: Ground truth analysis demonstrated that **91.4%** of true $S2$ and $S3$ siblings share identical building digits and core name tokens. When an S1 entity links to an S2 candidate with high confidence ($P \ge 0.88$), Version 2.1 inspects sub-threshold S3 candidates ($0.50 \le P < \tau^*$). If the S3 candidate agrees with the S2 sibling on structural address digits and clean tokens, it is dynamically promoted into the match set.
4. **Anti-Stealing Ambiguity Rejection**: To protect precision against the severe $4\times$ penalty of false positives under Macro $F_{0.5}$, when multiple S1 entities contest the same target candidate with probability gap $\delta < 0.05$, the system rejects the candidate from all claimants rather than gambling on a marginal $\arg\max$.

---

## 2. Methodology & Key Innovations

### 2.1 Problem Analysis
Exploratory data analysis revealed four core challenges:
1. **Severe Combinatorial Space**: The test set features 1,732,544 Source 1 records against ~10M Source 2 & 3 records, ruling out unconstrained quadratic comparisons.
2. **Asymmetric Error Penalties in $F_{0.5}$**:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Precision is prioritized $4\times$ over recall ($\beta^2 = 0.25$). Falsely merging non-identical entities degrades the metric four times faster than missing a marginal candidate.
3. **High Singleton Frequency & The Singleton Trap**: In ground truth analysis, ~5.6% of Source 1 entities have zero matching records. Correctly predicting an empty set yields an immediate $1.0$, whereas predicting a single speculative false merge collapses the entity's score straight to $0.0$.
4. **Domain Shift & Multilingual Scripts**:
   - **India (46.75% of test data)**: Over 15% of business names are written in native scripts (Hindi, Tamil, Telugu, Gujarati, Bengali, Odia) while the reference S1 record is in English. However, 100% of these records retain identifiable Latin/numeric address components (plot numbers, building numbers, phone numbers, PIN codes).
   - **France (Zero-Shot domain)**: Frequent usage of city names (`Nantes`, `Bordeaux`, `Paris`, `Lyon`) within company trade names (`Nantes Lycée SASU`), requiring city-aware token handling to prevent false-merge attractors.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Channel Country-Partitioned Inverted Indexing + Dense 25-Feature Engineering + Cost-Sensitive XGBoost + Sibling Triangulation + Anti-Stealing Bipartite Conflict Resolution.
- **Core Innovations**:
  - **Script-Preserving Normalization**: Preserves Unicode glyphs for Indic languages while normalizing Latin diacritics.
  - **Precomputed IDF-Damped Multi-Key Blocking**: Direct indexing of house number + street name, complex plot codes (`WZ-187C`, `6-2-101`), `c/o` person names, phone numbers, and compound domain stems, elevating blocker recall ceiling to 90.35%.
  - **Global Target Exclusivity & Anti-Stealing**: Enforces 1-to-at-most-1 target assignment globally, preventing multi-entity collision spikes while rejecting ambiguous ties ($\delta < 0.05$).

---

## 3. Candidate Generation (Blocking)

- **Blocking Channels Used**:
  - `ex`: Standardized clean business name (length $\ge 3$).
  - `bi`: First two distinctive name tokens (e.g., `novent_owl`).
  - `comp`: Compound concatenation of the first two tokens (e.g., `hermantotal` for matching `hermantotal.com`).
  - `tk`: Informative distinctive tokens ($\ge 3$ characters, skipping standalone French city stopwords).
  - `addr_st`: Address house/building digits combined with clean alphabetic street words (normalized for hyphens and slashes).
  - `code`: Complex plot, shop, and sector codes (e.g., `wz187c`, `af684`, `62101`, `59/101`).
  - `addr_co`: `c/o` (care of) personal name tokens in India (e.g., `gurnav_singh`).
  - `post_num`: Postal code concatenated with house number.
  - `phone`: 10-digit standardized contact numbers embedded in business name or address strings.
- **Precomputed IDF Attenuation**: Buckets up to 350 items are retained with logarithmic IDF weighting, while overgrown noise terms ($> 350$) are pruned.
- **Candidate Pool Constrained**: Top 18 candidates per Source 1 entity.
- **Validation Recall**: Measured at **90.35%** on out-of-fold validation.

---

## 4. Matching Model

### 4.1 Feature Engineering (25 Dense Features)
- **Name Similarities**:
  - `name_exact`: Exact match boolean on normalized stems.
  - `name_containment`: Substring inclusion boolean.
  - `name_tok_jaccard`: Word-level token Jaccard overlap.
  - `name_char_jaccard`: Character 3-gram Jaccard similarity.
  - `name_prefix`: Common prefix length ratio.
  - `name_len_diff`: Absolute length difference ratio.
  - `name_dice_2g`: Character 2-gram Dice coefficient.
  - `first_tok_match`: Exact agreement on the primary distinctive name token.
  - `token_overlap_ratio`: Recall ratio of S1 tokens covered in candidate tokens.
  - `domain_stem_match`: Boolean indicating whether S1 tokens are contained inside target domain/compound strings.
- **Address & Spatial Features**:
  - `num_match`: Building/house number concordance (1.0 for match, 0.5 for missing/neutral, 0.0 for conflict).
  - `postal_match`: Postal code agreement (1.0 match, 0.5 neutral, 0.0 conflict).
  - `addr_tok_jaccard`: Standardized address token overlap.
  - `addr_char_jaccard`: Character 3-gram address similarity.
  - `street_words_jaccard`: Jaccard similarity over alphabetic street names excluding digits and noise words.
  - `pure_address_match`: High-precision address boolean (numbers match + address Jaccard $\ge 0.65$).
  - `code_match`: Complex plot/shop code concordance boolean.
  - `postal_mismatch_veto`: 1.0 if postal codes are both present and disagree.
  - `house_num_mismatch_veto`: 1.0 if house numbers are both present and disagree.
- **Meta, Contact & Interaction Signals**:
  - `phone_match`: Contact number agreement (1.0 match, 0.5 neutral/missing, 0.0 conflict).
  - `has_non_ascii`: Boolean flag for non-Latin target names, allowing the model to adaptively trust pure address concordances for translated records.
  - `is_s2` / `is_s3`: Source origin indicators.
  - `rank_feat`: Retrieval rank within candidate blocking stage.
  - `name_addr_interaction`: Product of name and address character similarities.

### 4.2 Model Type & Training
- **Model**: Gradient Boosted Decision Trees (`XGBoost` v3.4.0, Apache 2.0).
- **Hyperparameters**: `n_estimators=350`, `max_depth=6`, `learning_rate=0.07`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=0.35` (cost-sensitive false positive penalty), objective: `binary:logistic`.
- **Validation**: Grouped K-Fold split strictly by `source1_entity_id` to guarantee zero data leakage between training and validation entities.
- **Decision Engine (Version 2.1 Tuned Parameters)**:
  - $\tau^* = 0.720$: Base probability threshold for candidate acceptance.
  - $\tau_{\text{singleton}} = 0.740$: Singleton protection shield.
  - $\Delta = 0.180$: Maximum allowable probability margin from top candidate.
  - $\delta_{\text{ambiguity}} = 0.050$: Anti-stealing minimum gap threshold.

---

## 5. Progression & Validation Results

| Pipeline Version | Key Innovations | Candidate Recall | Target Collisions | Validation Macro $F_{0.5}$ |
|---|---|---|---|---|
| **v1.0 (Initial)** | ASCII regex stripping, unconstrained argmax, no singleton shield | 84.1% | 258,426 collisions | 0.559 (Leaderboard collapse) |
| **v2.0 (Baseline)** | Multi-channel blocking, 20 features, 3-tier filter, Bipartite resolver | 88.3% | 0 (Strictly 1-to-1) | 0.9157 (91.6%) |
| **v2.1 (Advanced)** | IDF damping, 25 features, S2/S3 triangulation, Anti-Stealing rejection | **90.35%** | **0 (Strictly 1-to-1)** | **0.9344 (93.44%)** |

---

## 6. Code Artefacts & Structure

The complete self-contained package is located in `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multilingual normalization & tokenization.
- `src/blocking.py`: Version 2.0 candidate generation engine.
- `src/blocking_v21.py`: Version 2.1 IDF-damped candidate generation engine.
- `src/features.py`: Version 2.0 20-feature extractor.
- `src/features_v21.py`: Version 2.1 25-feature extractor.
- `src/model.py`: Version 2.0 XGBoost model & bipartite resolver.
- `src/model_v21.py`: Version 2.1 GBDT with S2/S3 triangulation & anti-stealing.
- `src/fast_inference.py`: Version 2.0 multi-country inference engine.
- `src/fast_inference_v21.py`: Version 2.1 multi-country inference engine.
- `src/evaluate.py`: Macro $F_{0.5}$ evaluation metric.
- `requirements.txt`: Pinned dependencies.
