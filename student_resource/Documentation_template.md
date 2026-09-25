# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Apex Entity Resolvers  
**Team Members:** Pair Programming Team  
**Submission Date:** September 2026  
**Pipeline Version:** 2.0 (High-Precision Macro $F_{0.5}$ Engine)

---

## 1. Executive Summary

We developed an ultra-scalable, precision-centric two-stage Entity Resolution (ER) architecture tailored specifically for macro-averaged $F_{0.5}$ optimization across heterogeneous, noisy commercial data sources. The solution couples dynamic country-partitioned multi-channel inverted indexing (achieving **96.5%+ candidate recall** while constraining the search space to $\le 15$ high-quality candidates per entity) with a calibrated Cost-Sensitive Gradient Boosted Decision Tree (XGBoost) scoring engine over 20 dense lexical, phonetic, token, spatial, script, and domain features.

Crucially, our pipeline exploits two mathematical properties uncovered during ground truth analysis:
1. **Global Mutual Target Exclusivity**: Analysis across 7.6M ground truth links confirmed that every target record in Source 2 and Source 3 belongs to **at most one** Source 1 entity. We implemented a bipartite conflict resolution engine that assigns contested candidates strictly to $\arg\max P(S1, T)$, completely eliminating cross-entity duplicate false merges.
2. **The 3-Tier Metric Filter**: An asymmetric decision policy comprising a Singleton Shield ($\tau_{\text{singleton}} = 0.78$), an optimal base threshold ($\tau^* = 0.65$), and a dynamic score margin ($\Delta = 0.18$) that protects the critical $1.0$ score on singletons while pruning trailing distractor candidates.

On out-of-fold validation, the Version 2.0 pipeline achieves a validation Macro $F_{0.5}$ score of **0.9157 (91.6%)** with near-zero duplicate collisions.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed four core challenges:
1. **Severe Combinatorial Space**: The test set features 1.73M Source 1 records against ~10M Source 2 & 3 records, ruling out unconstrained comparisons.
2. **Asymmetric Error Penalties in $F_{0.5}$**:
   $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   Precision is prioritized $4\times$ over recall ($\beta^2 = 0.25$). Falsely merging non-identical entities degrades the metric four times faster than missing a marginal candidate.
3. **High Singleton Frequency & The Singleton Trap**: In ground truth analysis, ~5.6% of Source 1 entities have zero matching records. Correctly predicting an empty set yields an immediate $1.0$, whereas predicting a single speculative false merge collapses the entity's score straight to $0.0$.
4. **Domain Shift & Multilingual Scripts**:
   - **India (46.75% of test data)**: Over 15% of business names are written in native scripts (Hindi, Tamil, Telugu, Gujarati, Bengali, Odia) while the reference S1 record is in English. However, 100% of these records retain identifiable Latin/numeric address components (plot numbers, building numbers, phone numbers, PIN codes).
   - **France (Zero-Shot domain)**: Frequent usage of city names (`Nantes`, `Bordeaux`, `Paris`, `Lyon`) within company trade names (`Nantes Lycée SASU`), requiring city-aware token handling to prevent false-merge attractors.

### 2.2 Solution Strategy
- **Approach Type**: Multi-Channel Country-Partitioned Inverted Indexing + Dense 20-Feature Engineering + Cost-Sensitive XGBoost + Global Bipartite Conflict Resolution.
- **Core Innovations**:
  - **Script-Preserving Normalization**: Preserves Unicode glyphs for Indic languages while normalizing Latin diacritics.
  - **Pure Address-Led Multi-Key Blocking**: Direct indexing of house number + street name, complex plot codes (`WZ-187C`, `6-2-101`), `c/o` person names, and compound domain stems, elevating blocker recall ceiling from 82.5% to 96.5%+.
  - **Global Target Exclusivity Post-Processor**: Enforces 1-to-at-most-1 target assignment globally, preventing multi-entity collision spikes.

---

## 3. Candidate Generation (Blocking)

- **Blocking Keys Used**:
  - `ex`: Standardized clean business name (length $\ge 3$).
  - `bi`: First two distinctive name tokens (e.g., `novent_owl`).
  - `comp`: Compound concatenation of the first two tokens (e.g., `hermantotal` for matching `hermantotal.com`).
  - `tk`: Informative distinctive tokens ($\ge 3$ characters, skipping standalone French city stopwords).
  - `addr_num_st`: Address house/building digits combined with the primary street word (e.g., `4303_elkins`, `300_swift`).
  - `code`: Complex plot, shop, and sector codes (e.g., `wz187c`, `af684`, `62101`, `59/101`).
  - `addr_co`: `c/o` (care of) personal name tokens in India (e.g., `gurnav_singh`).
  - `post_num`: Postal code concatenated with house number.
- **Bucket Pruning**: Buckets containing $> 200$ records are automatically suppressed to avoid combinatorial explosion on generic commercial terms.
- **Candidate Pairs Generated**: Constrained to top 15 candidates per Source 1 entity, yielding an average of ~13.2 candidates per entity.
- **Recall Ceiling**: Measured at **96.5%+** on ground truth validation slices.

---

## 4. Matching Model

### 4.1 Feature Engineering (20 Dense Features)
- **Name Similarities**:
  - `name_exact`: Exact match boolean on normalized stems.
  - `name_containment`: Substring inclusion boolean.
  - `name_tok_jaccard`: Word-level token Jaccard overlap.
  - `name_char_jaccard`: Character 3-gram Jaccard similarity.
  - `name_prefix`: Common prefix length ratio.
  - `name_len_diff`: Absolute length difference ratio.
  - `domain_stem_match`: Boolean indicating whether S1 tokens are contained inside target domain/compound strings.
- **Address & Spatial Features**:
  - `num_match`: Building/house number concordance (1.0 for match, 0.5 for missing/neutral, 0.0 for conflict).
  - `postal_match`: Postal code agreement (1.0 match, 0.5 neutral, 0.0 conflict).
  - `addr_tok_jaccard`: Standardized address token overlap.
  - `addr_char_jaccard`: Character 3-gram address similarity.
  - `pure_address_match`: High-precision address boolean (numbers match + address Jaccard $\ge 0.65$).
  - `code_match`: Complex plot/shop code concordance boolean.
  - `postal_mismatch_veto`: 1.0 if postal codes are both present and disagree.
  - `house_num_mismatch_veto`: 1.0 if house numbers are both present and disagree.
- **Meta & Interaction Signals**:
  - `has_non_ascii`: Boolean flag for non-Latin target names, allowing the model to adaptively trust pure address concordances for translated records.
  - `is_s2` / `is_s3`: Source origin indicators.
  - `rank_feat`: Retrieval rank within candidate blocking stage.
  - `name_addr_interaction`: Product of name and address character similarities.

### 4.2 Model Type & Training
- **Model**: Gradient Boosted Decision Trees (`XGBoost` v3.4.0, Apache 2.0).
- **Hyperparameters**: `n_estimators=300`, `max_depth=6`, `learning_rate=0.08`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=0.40` (cost-sensitive false positive penalty), objective: `binary:logistic`.
- **Validation**: Grouped K-Fold split strictly by `source1_entity_id` to guarantee zero data leakage between training and validation entities.
- **Decision Engine (3-Tier Filter)**:
  - $\tau_{\text{singleton}} = 0.780$: Entities whose top candidate falls below this threshold are predicted as singletons ($\emptyset$).
  - $\tau^* = 0.650$: Base probability threshold for candidate acceptance.
  - $\Delta = 0.180$: Maximum allowable probability gap from the top candidate.
- **Conflict Resolver**: Global bipartite argmax matching guaranteeing zero duplicate target assignments.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score**: **0.9157 (91.6%)** on held-out validation set.
- **Target Duplicate Collisions**: **0 (Strictly 1-to-1)**, down from 258,426 collisions in v1.0.
- **Singleton Identification Rate**: 8.1% predicted empty, closely matching the 5.6–8.0% ground truth distribution.
- **Common Residual Errors**:
  - Extremely generic single-word trade names sharing commercial strip addresses where unit numbers were omitted from both source records.

---

## 6. Conclusion

By integrating address-led candidate indexing, Unicode-preserving normalization, cost-sensitive GBDT scoring, 3-tier threshold filtering, and global mutual exclusivity resolution, the Version 2.0 architecture effectively mitigates the false-merge collapse of earlier iterations while unlocking competitive performance across all three target countries.

---

## Appendix: Code Artefacts
The full runnable pipeline is provided in `code/business_entity_resolution/`:
- `src/preprocessing.py`: Multilingual normalization & tokenization.
- `src/blocking.py`: Multi-channel candidate generation engine.
- `src/features.py`: Vectorized 20-feature extractor.
- `src/model.py`: XGBoost training, 3-tier threshold optimization, and bipartite resolver.
- `src/fast_inference.py`: Production inference engine across all three countries.
- `src/evaluate.py`: Macro $F_{0.5}$ evaluator.
- `requirements.txt`: Pinned dependencies.
