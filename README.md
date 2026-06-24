# Redrob Hackathon v4 — Team Antigravity
## Retrieval / Recommendation Systems Engineer Ranking System

This repository contains Team Antigravity's final submission for the Redrob Hackathon v4. We have designed, developed, and optimized an automated candidate ranking pipeline capable of processing 100,000 raw candidate resumes and algorithmically isolating the top 100 best matches for a highly specialized "Retrieval / Recommendation Systems Engineer" role.

---

### 1. Architectural Approach & Overview

The system processes candidate data (in JSONL format) through a multi-stage funnel. Rather than relying entirely on black-box LLM calls or simplistic keyword matching, our pipeline employs a robust **three-signal ensemble architecture**:

1. **Deterministic Feature Engineering (50% Weight)**: We extract hard facts from candidate profiles—such as whether they have built vector databases in production environments, their history at product-centric companies versus outsourced consulting firms, and their experience with ranking evaluation metrics (NDCG, MRR).
2. **Dense Semantic Embeddings (30% Weight)**: Using `all-MiniLM-L6-v2` SentenceTransformers, we encode candidate resumes and the specific Job Description (JD) into dense vectors to measure deep semantic similarity. This captures latent conceptual overlap (e.g., matching "ANN search" to "Vector Databases").
3. **Sparse BM25 Keyword Overlap (20% Weight)**: We index candidate profiles and headlines using the BM25 algorithm to guarantee exact keyword matches against critical JD terms.

By biasing the ensemble heavily towards deterministic rules (50%), we prevent candidates with purely superficial vocabulary from floating to the top, ensuring only hands-on, production-level builders are ranked highly.

---

### 2. Detailed Pipeline Execution Flow

The system is broken down into modular, sequential scripts to ensure scalability and memory efficiency. 

- **`precompute/01_parse_candidates.py`** 
  Reads the raw 100,000 JSONL candidates line-by-line. Normalizes the complex nested schemas, handles missing data safely, and dumps an intermediate parsed dataset. *(Approximate CPU Runtime: 2 mins)*
  
- **`precompute/02_extract_features.py`** 
  The core business-logic engine. Maps the JD requirements into strict Python logic. It evaluates 8 different feature groups (A-H), calculates base scores, applies soft multipliers, and evaluates all 6 hard disqualifier rules. Outputs `feature_matrix.pkl`. *(Approximate CPU Runtime: 2 mins)*
  
- **`precompute/03_build_bm25.py`** 
  Constructs a sparse BM25 index utilizing candidate summaries, headlines, and career histories. Computes exact token frequency scores against a tailored set of JD keywords. *(Approximate CPU Runtime: 3 mins)*
  
- **`precompute/04_build_embeddings.py`** 
  Uses the `sentence-transformers` library to build 384-dimensional dense vectors for all 100,000 candidates. This process is batch-optimized and utilizes PyTorch. *(Approximate CPU Runtime: 50-60 mins)*
  
- **`rank.py`** 
  The final aggregation script. It calculates cosine similarities from the embeddings, merges them with BM25 and feature scores using the ensemble formula, and outputs the definitive top 100 candidates to `outputs/team_antigravity.csv`.
  
- **`generate_reasoning.py`** 
  A utility module called dynamically by `rank.py`. It statically generates completely transparent, factual reasoning strings for every top candidate so human recruiters can understand exactly *why* a candidate received their score.
  
- **`validate_submission.py`** 
  Ensures the final CSV contains exactly 100 rows, validates candidate IDs, and checks data types for compliance with hackathon standards.

---

### 3. Comprehensive Scoring Mechanics

#### The Ensemble Formula
`Final Score = (0.50 * Feature Score) + (0.30 * Embedding Score) + (0.20 * BM25 Score)`

#### The Base Feature Score
`Base = (0.30 * Production Retrieval) + (0.20 * Product Company) + (0.15 * Eval Depth) + (0.15 * Domain Fit) + (0.10 * Experience) + (0.05 * Location) + (0.05 * Engagement)`
*(The base score is capped at 1.0 after adding up to +0.05 Preferred Bonus, before soft multipliers are applied.)*

#### Hard Disqualifiers (Score = 0.0)
We mercilessly drop candidates if they trigger any of the following 6 rules:
1. `is_honeypot`: Internal platform flag for synthetic/spam profiles.
2. `is_consulting_only`: Candidates who have spent 100% of their career exclusively at IT services/consulting firms (TCS, Infosys, Wipro, etc.).
3. `is_pure_research`: Candidates holding purely academic titles (PhD/Postdoc/Researcher) who have zero evidence of deploying models to production.
4. `is_llm_wrapper_only`: Candidates with < 12 months AI experience who lack foundational pre-LLM ML signals (e.g., PyTorch, TensorFlow).
5. `is_not_coding_ic`: Candidates who have been in management roles (e.g., Director, VP) for more than 18 months, indicating they are no longer hands-on contributors.
6. `is_wrong_domain`: Candidates whose background is purely Computer Vision or Speech processing with zero NLP/IR crossover.

#### Soft Penalizers & Multipliers
- `is_title_chaser`: Candidates with more than 3 recent extremely short job stints (<18 months). Penalty: `* 0.75`
- `is_framework_only`: Candidates whose skills are dominated by high-level frameworks (Langchain, Streamlit) without underlying architectural depth (Attention, Loss Functions). Penalty: `* 0.80`
- `is_closed_source`: Seniors (5+ years) with zero evidence of Open Source, papers, or technical talks. Penalty: `* 0.85`
- `recruiter_response_rate`: If historical recruiter response rate < 10%. Penalty: `* 0.85`
- `notice_multiplier`: Scales linearly based on notice period. `1.0` for 30 days or less, dropping down to `0.55` for 120+ days.
- `recency_multiplier`: Scales based on the last active date. `1.0` if active in the last 90 days, dropping to `0.70` if inactive for over 180 days.

---

### 4. JD Analysis & Feature Group Logic

All deterministic keyword lists were meticulously derived directly from the provided JD.

- **RETRIEVAL_TECH**: (pinecone, qdrant, weaviate, faiss) — Vital because the JD asks for hands-on vector database architectures.
- **EVAL_KEYWORDS**: (ndcg, mrr, a/b test, offline evaluation) — Essential because the JD heavily emphasizes "strong evaluation fundamentals" and measuring offline-to-online drift.
- **NLP_IR_KEYWORDS**: Crucial to isolate dedicated ranking and semantic search expertise from broader, generic AI fields (like Vision or Audio).

We also extract bonus signals including LoRA fine-tuning, Learning-to-Rank (LTR), HR Tech experience, and verified GitHub activity.

---

### 5. Final Top 10 Candidate Results

| Rank | Candidate ID | Final Score | Feature Score | Embedding Score | BM25 Score | Title | Experience | Location | Notice Period |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CAND_0081846 | 0.7808 | 0.9419 | 0.4170 | 0.9235 | Lead AI Engineer | 6.7 yrs | Jaipur, Rajasthan | 30 days |
| 2 | CAND_0055905 | 0.767 | 0.9083 | 0.4737 | 0.8534 | Senior Machine Learning Engineer | 8.1 yrs | London | 30 days |
| 3 | CAND_0086022 | 0.7482 | 0.8501 | 0.4104 | 1.0000 | Senior Applied Scientist | 5.3 yrs | Kolkata, West Bengal | 0 days |
| 4 | CAND_0079387 | 0.7426 | 0.8742 | 0.5226 | 0.7435 | AI Engineer | 6.9 yrs | Trivandrum, Kerala | 30 days |
| 5 | CAND_0077337 | 0.7369 | 0.8214 | 0.4873 | 0.9001 | Staff Machine Learning Engineer | 7.0 yrs | Kochi, Kerala | 60 days |
| 6 | CAND_0075249 | 0.7138 | 0.7479 | 0.5683 | 0.8470 | Applied ML Engineer | 6.2 yrs | Ahmedabad, Gujarat | 45 days |
| 7 | CAND_0096172 | 0.7088 | 0.7402 | 0.5850 | 0.8160 | NLP Engineer | 5.2 yrs | Chennai, Tamil Nadu | 45 days |
| 8 | CAND_0046525 | 0.7083 | 0.7650 | 0.4817 | 0.9062 | Senior Machine Learning Engineer | 6.1 yrs | Pune, Maharashtra | 60 days |
| 9 | CAND_0030953 | 0.7041 | 0.7430 | 0.5345 | 0.8612 | Search Engineer | 7.8 yrs | Chennai, Tamil Nadu | 45 days |
| 10 | CAND_0062247 | 0.7039 | 0.7374 | 0.6145 | 0.7542 | AI Engineer | 7.3 yrs | Kochi, Kerala | 30 days |

---

### 6. Score Distributions & Statistics

*Score Distribution across the final top 100 ranked candidates:*
- **p50**: 0.6091
- **p75**: 0.6558
- **p90**: 0.6972
- **p95**: 0.7150
- **p99**: 0.7671
- **p100**: 0.7808


---

### 7. Design Tradeoffs & What We Ignored

**Why consulting-only candidates are hard-disqualified:**
The JD heavily implies a need for a high-autonomy "0-to-1" builder who has owned product-level metrics and shipped core retrieval infrastructure. Culturally, this aligns closer to product-focused environments than outsourced services. Therefore, candidates lacking product-company experience were entirely dropped.

**What the pipeline deliberately ignores:**
The pipeline intentionally ignores education (degrees and universities), certifications, compensation history, and spoken languages. We concluded that culture fit, hands-on production engineering with vector databases, and evaluation fundamentals are far more critical indicators of success for this specific role than academic pedigree or paper certificates.

---

### 8. System Limitations

- **Hardcoded Logic**: The system relies heavily on explicit text lists tailored exactly to this JD. It will not generalize to a "Frontend Engineer" or "Data Analyst" role without entirely new feature logic.
- **Approximate Timeframes**: Because `start_date` and `end_date` are not reliably standardized, our logic approximating durations (e.g., checking if a candidate is a "Title Chaser" based on recent short stints) is heuristic.
- **Semantic Domain Misses**: Subtle domain mismatches might slip through if the candidate describes unrelated technical fields using dense NLP terminology.

---

### 9. How to Reproduce

To execute the full pipeline from scratch, run the following commands sequentially inside a Python virtual environment:

```bash
pip install -r requirements.txt
python precompute/01_parse_candidates.py
python precompute/02_extract_features.py
python precompute/03_build_bm25.py
python precompute/04_build_embeddings.py
python rank.py
python validate_submission.py
```
*Note: `04_build_embeddings.py` generates vectors for all 100,000 candidates and requires approximately 50-60 minutes to complete on a standard CPU. Large generated data artifacts (`.jsonl`, `.npy`, `.pkl`) should be `.gitignore`d.*
