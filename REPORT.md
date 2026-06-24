# Redrob Hackathon v4 — Team Antigravity
## Candidate Ranking System Report

### 1. Approach Summary
The pipeline processes 100,000 raw candidate resumes to find the top 100 matches for a specialized "Retrieval / Recommendation Systems Engineer" role. It parses candidate JSON lines, extracts hard disqualifiers and soft penalizer flags based on explicit job description constraints, and computes both semantic similarity (via SentenceTransformers) and keyword overlap (via BM25). The final candidate ranking relies on a three-signal ensemble composed of 50% deterministic feature-based scores (capturing domain fit, evaluation depth, and product-company background), 30% semantic embedding similarity against the JD, and 20% exact keyword matching.

### 2. Pipeline Architecture
- **precompute/01_parse_candidates.py**: Parses JSONL candidate data and normalizes the schema (~2 mins CPU).
- **precompute/02_extract_features.py**: Generates the `feature_matrix.pkl` by applying feature groups and scoring rules to each candidate (~2 mins CPU).
- **precompute/03_build_bm25.py**: Builds a BM25 index over candidate profiles/headlines and scores against JD keywords (~3 mins CPU).
- **precompute/04_build_embeddings.py**: Encodes candidate texts and JD using `all-MiniLM-L6-v2` to compute semantic vectors (~50-60 mins CPU).
- **rank.py**: Joins feature scores, BM25 scores, and embeddings to compute the final ensemble score, caps output at the top 100, and generates `team_antigravity.csv` (~1 min CPU).
- **generate_reasoning.py**: Utility module called by `rank.py` to statically generate transparent score breakdowns and matching justifications for the top candidates.
- **validate_submission.py**: Validates the final submission CSV schema, row count, and data integrity.
- **sandbox/app.py**: A local UI test app to visually inspect and debug the ranking output.

### 3. Scoring Formula
**Ensemble Formula**:
`Final Score = (0.50 * Feature Score) + (0.30 * Embedding Score) + (0.20 * BM25 Score)`

**Base Feature Score Formula**:
`Base = (0.30 * Production Retrieval) + (0.20 * Product Company) + (0.15 * Eval Depth) + (0.15 * Domain Fit) + (0.10 * Experience) + (0.05 * Location) + (0.05 * Engagement)`
*(The base score is then capped at 1.0 after adding up to +0.05 Preferred Bonus, before soft multipliers are applied.)*

**Hard Disqualifiers (Score = 0.0)**:
1. `is_honeypot`: Discards synthetic or spam candidates flagged by the platform.
2. `is_consulting_only`: Discards candidates who have spent 100% of their career exclusively at IT consulting firms (e.g., TCS, Infosys).
3. `is_pure_research`: Discards PhD/Postdoc researchers who have never deployed or shipped models to production.
4. `is_llm_wrapper_only`: Discards candidates with < 12 months AI experience who lack foundational pre-LLM ML signals (e.g., PyTorch, TensorFlow).
5. `is_not_coding_ic`: Discards candidates who have been in management roles (e.g., Director, VP) for more than 18 months.
6. `is_wrong_domain`: Discards candidates whose background is purely Computer Vision or Speech with zero NLP/IR crossover.

**Soft Penalizers & Multipliers**:
- `is_title_chaser`: `* 0.75` multiplier.
- `is_framework_only`: `* 0.80` multiplier.
- `is_closed_source`: `* 0.85` multiplier.
- `recruiter_response_rate < 0.10`: `* 0.85` multiplier.
- `notice_multiplier`: Ranges from `1.0` (<=30 days) to `0.55` (>120 days).
- `recency_multiplier`: Ranges from `1.0` (<=90 days inactive) to `0.70` (>180 days inactive).

### 4. Feature Groups
- **Group A (Production Retrieval)**: Weight 0.30. Captures whether the candidate has built or shipped vector databases and retrieval systems in a production environment.
- **Group B (Product Company Score)**: Weight 0.20. Captures the fraction of their career spent at product-focused companies rather than outsourced IT/academic roles.
- **Group C (Eval Depth)**: Weight 0.15. Captures whether they have experience rigorously evaluating ranking models with metrics like NDCG or MRR.
- **Group D (Domain Fit)**: Weight 0.15. Captures their overlap with NLP and Information Retrieval, actively penalizing pure CV/Speech backgrounds.
- **Group E (Experience Range)**: Weight 0.10. Captures whether their years of experience falls into the ideal mid-senior band for this role.
- **Group F (Location)**: Weight 0.05. Captures whether their location aligns with the preferred hubs (Pune, Noida) or acceptable alternatives.
- **Group G (Engagement)**: Weight 0.05. Captures their recent activity, notice period, platform response rate, and open-to-work flag.
- **Group H (Preferred Bonus)**: Weight +0.05. Captures nice-to-have signals like LoRA fine-tuning, Learning-to-Rank, HR Tech experience, and high GitHub activity.

### 5. JD Analysis Decisions
All keyword lists were manually derived from the JD and hardcoded into the feature extraction logic. 
- **RETRIEVAL_TECH** (pinecone, qdrant, weaviate, faiss): Vital because the JD specifically asks for hands-on vector database and semantic search implementations.
- **EVAL_KEYWORDS** (ndcg, mrr, a/b test, offline evaluation): Essential because the JD heavily emphasizes "strong evaluation fundamentals" and measuring offline-to-online drift.
- **NLP_IR_KEYWORDS**: Crucial to isolate dedicated ranking and semantic search expertise from broader, generic AI fields (like Vision or Audio).

### 6. Top 10 Results
| Rank | Candidate ID | Final Score | Feature Score | Embedding Score | BM25 Score | Title | Experience | Location | Notice Period |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CAND_0081846 | 0.7808 | 0.9419 | 0.4170 | 0.9235 | N/A | 6.7 yrs | Jaipur, Rajasthan | N/A days |
| 2 | CAND_0055905 | 0.7670 | 0.9083 | 0.4737 | 0.8534 | N/A | 8.1 yrs | London | N/A days |
| 3 | CAND_0086022 | 0.7482 | 0.8501 | 0.4104 | 1.0000 | N/A | 5.3 yrs | Kolkata, West Bengal | N/A days |
| 4 | CAND_0079387 | 0.7426 | 0.8742 | 0.5226 | 0.7435 | N/A | 6.9 yrs | Trivandrum, Kerala | N/A days |
| 5 | CAND_0077337 | 0.7369 | 0.8214 | 0.4873 | 0.9001 | N/A | 7.0 yrs | Kochi, Kerala | N/A days |
| 6 | CAND_0075249 | 0.7138 | 0.7479 | 0.5683 | 0.8470 | N/A | 6.2 yrs | Ahmedabad, Gujarat | N/A days |
| 7 | CAND_0096172 | 0.7088 | 0.7402 | 0.5850 | 0.8160 | N/A | 5.2 yrs | Chennai, Tamil Nadu | N/A days |
| 8 | CAND_0046525 | 0.7083 | 0.7650 | 0.4817 | 0.9062 | N/A | 6.1 yrs | Pune, Maharashtra | N/A days |
| 9 | CAND_0030953 | 0.7041 | 0.7430 | 0.5345 | 0.8612 | N/A | 7.8 yrs | Chennai, Tamil Nadu | N/A days |
| 10 | CAND_0062247 | 0.7039 | 0.7374 | 0.6145 | 0.7542 | N/A | 7.3 yrs | Kochi, Kerala | N/A days |

### 7. Score Distribution
*Distribution across the final top 100 ranked candidates:*
- **p50**: 0.6091
- **p75**: 0.6558
- **p90**: 0.6972
- **p95**: 0.7150
- **p99**: 0.7671
- **p100**: 0.7808

*Disqualification statistics across all 100,000 candidates:*
- **Total Disqualified (Score = 0.0)**: 74,094
  - `is_honeypot`: 17,355
  - `is_consulting_only`: 8,940
  - `is_pure_research`: 0
  - `is_llm_wrapper_only`: 64,568
  - `is_not_coding_ic`: 0
  - `is_wrong_domain`: 0

*(Note: Candidates can trigger multiple disqualifier flags simultaneously).*

### 8. Design Decisions and Tradeoffs

**Why feature rules carry 50% weight (not embeddings)**:
We deliberately biased the ensemble towards deterministic feature rules rather than semantic embeddings because semantic search tends to be overly forgiving and opaque. By allocating 50% of the final score to hardcoded domain logic, we ensure that candidates without real production retrieval experience or strong evaluation metrics cannot float to the top just by having broadly similar terminology in their resumes.

**Why consulting-only candidates are hard-disqualified**:
Rather than softly penalizing candidates from IT services firms, we chose to hard-disqualify anyone with 100% of their career in consulting. The JD heavily implies a need for a high-autonomy "0-to-1" builder who has owned product-level metrics and shipped core retrieval infrastructure, which culturally aligns closer to product-focused environments than outsourced services.

**What the pipeline deliberately ignores and why**:
The pipeline intentionally ignores education (degrees and universities), certifications, compensation history, and spoken languages. We concluded that culture fit, hands-on production engineering with vector databases, and evaluation fundamentals are far more critical indicators of success for this specific role than academic pedigree or paper certificates.

### 9. Known Limitations
- JD is fully hardcoded; system does not generalize to other JDs without rewriting feature lists.
- Temporal date constraints (5-year window for title chaser, pre-2022 LLM check) approximate due to missing start_date/end_date parsing.
- github_activity_score numeric value now read directly but text fallback also present.
- is_wrong_domain checks only catch explicit CV/speech vocabulary; subtle domain mismatches may pass.

### 10. How to Reproduce
To execute the full pipeline from scratch, run the following commands sequentially inside the project virtual environment:

```bash
/home/achyuta/My_Project/redrob-v4/.venv/bin/python precompute/01_parse_candidates.py
/home/achyuta/My_Project/redrob-v4/.venv/bin/python precompute/02_extract_features.py
/home/achyuta/My_Project/redrob-v4/.venv/bin/python precompute/03_build_bm25.py
/home/achyuta/My_Project/redrob-v4/.venv/bin/python precompute/04_build_embeddings.py
/home/achyuta/My_Project/redrob-v4/.venv/bin/python rank.py
/home/achyuta/My_Project/redrob-v4/.venv/bin/python validate_submission.py
```
*Note: `04_build_embeddings.py` generates vectors for all 100,000 candidates and requires approximately 50 minutes to complete on CPU.*
