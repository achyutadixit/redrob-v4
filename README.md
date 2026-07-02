# Redrob Hackathon v4 — Team Antigravity

## Intelligent Candidate Ranking System for Retrieval / Recommendation Engineers

An automated, multi-signal candidate ranking pipeline that processes **100,000 raw candidate resumes** and surfaces the **top 100** best-fit matches for a highly specialized *Retrieval / Recommendation Systems Engineer* role. Built with a three-signal ensemble architecture combining deterministic feature engineering, dense semantic embeddings, and sparse keyword matching.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.14 | All pipeline scripts |
| **Semantic Embeddings** | `sentence-transformers` (`all-MiniLM-L6-v2`) | 384-dim dense vector encoding for candidates and JD |
| **Sparse Retrieval** | `rank_bm25` | BM25 keyword overlap scoring |
| **Numerical Compute** | `numpy` | Cosine similarity, score aggregation, percentile stats |
| **Serialization** | `pickle`, `json` | Intermediate artifact storage |
| **Deep Learning Backend** | `PyTorch` (CPU) | Underlying engine for SentenceTransformers |
| **Validation** | Custom `validate_submission.py` | Schema and constraint checks for final output |
| **Debugging UI** | `sandbox/app.py` | Local visual inspection of rankings |

---

## Repository Structure

```text
redrob-v4/
├── data/
│   ├── candidates.jsonl              # Raw 100K candidate dataset (gitignored, ~465 MB)
│   └── job_description.txt           # The target JD used for all matching and scoring
├── outputs/
│   └── team_antigravity.csv          # Final ranked top 100 with score + reasoning per row
├── precompute/
│   ├── 01_parse_candidates.py        # Stage 1: Parse JSONL → structured objects + honeypot detection
│   ├── 02_extract_features.py        # Stage 2: Deterministic feature extraction & disqualifiers
│   ├── 03_build_bm25.py              # Stage 3: Build BM25 index & score against JD keywords
│   ├── 04_build_embeddings.py        # Stage 4: Encode all candidates into 384-dim vectors
│   └── artifacts/                    # Pre-computed intermediates tracked via Git LFS
│       ├── embeddings.npy            #   ↳ 384-dim candidate vectors (147 MB, LFS)
│       ├── jd_embedding.npy          #   ↳ 384-dim JD vector (4 KB, LFS)
│       ├── bm25_scores.pkl           #   ↳ BM25 score dict per candidate (2.3 MB, LFS)
│       ├── feature_matrix.pkl        #   ↳ Feature dicts for all candidates (17 MB, LFS)
│       └── honeypots_flagged.json    #   ↳ Flagged honeypot candidate IDs (1.2 MB, plain JSON)
│       # candidates_parsed.pkl is gitignored (398 MB) — regenerate with 01_parse_candidates.py
├── sandbox/
│   └── app.py                        # Local test UI for visual debugging
├── .gitattributes                    # Git LFS tracking rules for *.pkl and *.npy under precompute/
├── generate_reasoning.py             # Produces transparent, factual justification strings
├── rank.py                           # Ensemble aggregation → final CSV output
├── validate_submission.py            # Validates output against hackathon constraints
├── .gitignore                        # Excludes raw dataset and largest intermediate from version control
└── README.md
```

---

## ⚠️ Dataset & Artifact Availability

### Pre-computed Artifacts (Git LFS)

All derived artifacts — `embeddings.npy`, `jd_embedding.npy`, `bm25_scores.pkl`, `feature_matrix.pkl` — are committed to this repository and tracked via **Git LFS**. After cloning, run:

```bash
git lfs pull
```

This downloads the LFS-tracked files and lets you skip directly to `rank.py` without rerunning the 50–60 minute embedding step.

> **Requires Git LFS:** Install from https://git-lfs.com or via your package manager (`sudo apt install git-lfs`).

### Raw Candidate Dataset

The raw candidate resume dataset (`candidates.jsonl`, ~465 MB) contains proprietary candidate profiles and is excluded from this repository via `.gitignore`. To run the pipeline from scratch, place the `candidates.jsonl` file in the **parent directory** of the repository:

```bash
/some/path/
├── redrob-v4/            # This repository
└── candidates.jsonl      # Place the raw dataset here
```

`candidates_parsed.pkl` (398 MB) is also gitignored as it exceeds LFS free-tier storage limits. It is fully reproducible by running `python precompute/01_parse_candidates.py`.

---

## How the Pipeline Works

The pipeline runs as a sequence of 4 precompute stages followed by a final ranking and output step. Each stage reads from the previous stage's artifacts and writes its own, keeping memory usage bounded and allowing individual stages to be re-run independently.

```text
candidates.jsonl
      │
      ▼
┌─────────────────────┐
│  01_parse_candidates │  Parse JSONL, detect honeypots, normalize schema
│  → candidates_parsed │  
│    .pkl              │  
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  02_extract_features │  Apply 6 disqualifiers, 6 soft penalizers, 8 feature groups
│  → feature_matrix   │  
│    .pkl              │  
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  03_build_bm25      │  Tokenize profiles, build BM25 index, score vs JD
│  → bm25_scores.pkl  │  
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  04_build_embeddings │  Encode with all-MiniLM-L6-v2, save dense vectors
│  → embeddings.npy   │  
│  → jd_embedding.npy │  
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  rank.py            │  Cosine similarity + BM25 + features → ensemble score
│  + generate_        │  Factual reasoning per candidate
│    reasoning.py     │  
│  → team_antigravity │  
│    .csv             │  
└─────────────────────┘
```

**Total end-to-end runtime:** ~60 minutes on CPU (dominated by Stage 4 embedding generation).

---

## Candidate Data: Schema & How We Use It

Each candidate record in `candidates.jsonl` is a nested JSON object. Here is how we consume every relevant field:

### `profile` (Top-level candidate metadata)

| Field | Type | How We Use It |
|---|---|---|
| `years_of_experience` | float | Mapped to an experience band score (sweet spot: 5–9 years = 1.0) |
| `current_title` | string | Checked for management signals (`VP`, `Director`, `CTO`) to flag `is_not_coding_ic` |
| `location` | string | Scored against preferred (Pune/Noida), acceptable (Hyderabad/Mumbai/Delhi), and tier-1 cities |
| `summary` | string | Concatenated into `profile_text` for keyword scanning across all feature groups |
| `headline` | string | Concatenated into `profile_text`; also used as BM25 document input |

### `career_history` (Array of job objects)

| Field | Type | How We Use It |
|---|---|---|
| `company` | string | Checked against `CONSULTING_FIRMS` set (TCS, Infosys, Wipro…) for product-company scoring |
| `title` / `role` | string | Scanned for research-only titles; used in reasoning output |
| `description` | string | The richest text field — scanned for retrieval tech in production context, deployment verbs, eval keywords, and domain signals |
| `duration_months` | int | Summed for product-company fraction; checked for short stints (title chaser detection) |
| `industry` | string | Used to exclude `academic` / `lab` / `research_org` from product-company time |
| `is_current` | bool | Used with management title detection to calculate current management tenure |

### `skills` (Array of skill objects)

| Field | Type | How We Use It |
|---|---|---|
| `name` | string | Matched against retrieval tech lists, framework lists, and AI skill lists |
| `proficiency` | string | Used in honeypot detection (expert + 0 duration = suspicious) |
| `duration_months` | int | Summed for AI skill duration to detect `is_llm_wrapper_only`; used in honeypot detection |

### `education` (Array of education objects)

| Field | Type | How We Use It |
|---|---|---|
| `end_year` | int | Used **only** for honeypot detection: `grad_year + years_exp > current_year + 2` flags synthetic profiles |

> **Design Decision:** We deliberately do not use degree names, university rankings, or GPAs for scoring. The JD asks for a hands-on builder, not an academic pedigree.

### `redrob_signals` (Platform-provided behavioral signals)

This is the key differentiator in our pipeline. The `redrob_signals` object contains behavioral metadata from the Redrob platform itself — data that goes beyond what a traditional resume parser would extract.

| Signal | Type | How We Use It |
|---|---|---|
| `notice_period_days` | int | Converted to a `notice_multiplier` (1.0 for ≤30d, 0.90 for ≤60d, 0.80 for ≤90d, 0.70 for ≤120d, 0.55 for >120d). Also used in the engagement sub-score and displayed in reasoning output. |
| `last_active_date` | date string | Converted to `recency_multiplier` (1.0 if ≤90 days ago, 0.85 if ≤180 days, 0.70 if >180 days). Stale candidates are penalized because they may not be actively seeking. |
| `recruiter_response_rate` | float (0–1) | Directly feeds into the engagement sub-score (20% weight). If below 0.10, an additional `* 0.85` penalty is applied as a soft penalizer — candidates who historically ignore recruiters are deprioritized. |
| `willing_to_relocate` | bool | If `True`, the candidate receives the maximum location score (1.0) regardless of their current city. This effectively treats relocatable candidates as if they are in a preferred location. |
| `open_to_work_flag` | bool | Contributes 10% to the engagement sub-score. Actively job-seeking candidates get a small boost. |
| `github_activity_score` | int | If > 30 (or text fallback matches OSS keywords), contributes to the `preferred_bonus` (+0.01). Rewards candidates with verified open-source activity. |

> **Why signals matter:** Two candidates with identical resumes can score very differently based on their signals. A candidate with a 30-day notice period, high recruiter response rate, and active-within-90-days status will outscore a similarly skilled candidate who has a 120-day notice, hasn't been active in 6 months, and ignores recruiters. This reflects real-world hiring constraints.

---

## Scoring Mechanics in Detail

### The Three-Signal Ensemble

```
Final Score = (0.50 × Feature Score) + (0.30 × Embedding Score) + (0.20 × BM25 Score)
```

We deliberately weight features at 50% because they encode hard, interpretable facts. Embeddings are powerful but opaque and tend to be overly forgiving of superficially similar text. BM25 acts as a precision anchor ensuring critical JD keywords are literally present.

If any hard disqualifier fires, the feature score is `0.0` and the entire final score becomes `0.0` — embeddings and BM25 cannot rescue a disqualified candidate.

### Feature Score Composition

```
Base = (0.30 × Production Retrieval)
     + (0.20 × Product Company)
     + (0.15 × Eval Depth)
     + (0.15 × Domain Fit)
     + (0.10 × Experience Range)
     + (0.05 × Location)
     + (0.05 × Engagement)

Base = min(Base + Preferred Bonus, 1.0)
Feature Score = Base × Π(soft multipliers)
```

#### Feature Group Details

| Group | Weight | What It Measures | Key Signals |
|---|---|---|---|
| **A: Production Retrieval** | 30% | Has the candidate built and deployed vector databases or retrieval systems in production? | Checks career descriptions for retrieval tech (Pinecone, FAISS, Qdrant…) co-occurring with deployment verbs (`built`, `shipped`, `deployed`). Also checks for hybrid search, embedding drift monitoring, and operational signals. |
| **B: Product Company** | 20% | What fraction of their career was spent at product-focused companies vs. consulting/outsourcing? | Excludes TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, Tech Mahindra, Mphasis, Hexaware, L&T Infotech. Also checks for evidence of shipping to real users. |
| **C: Eval Depth** | 15% | Does the candidate understand ranking evaluation rigorously? | Scans for NDCG, MRR, MAP, A/B testing, offline/online evaluation, precision@k, recall@k, click-through rate. Tiered: 0 matches = 0.0, 1–2 = 0.5, 3+ = 1.0. |
| **D: Domain Fit** | 15% | Is their primary domain NLP/IR, or is it CV/Speech? | Counts keyword hits across NLP/IR vs. CV vs. Speech. Pure NLP/IR = 1.0, mixed = 0.8, unknown = 0.5, pure CV/Speech = 0.2. |
| **E: Experience Range** | 10% | Are they in the sweet spot for this role? | <3 yrs = 0.2, 3–5 yrs = 0.5, **5–9 yrs = 1.0**, 9–12 yrs = 0.8, >12 yrs = 0.6. |
| **F: Location** | 5% | Are they in a preferred hiring hub? | Pune/Noida or willing to relocate = 1.0, Hyderabad/Mumbai/Delhi = 0.9, Bangalore/Chennai/Kolkata = 0.8, elsewhere = 0.5. |
| **G: Engagement** | 5% | Are they actively job-seeking and responsive? | Composite of recency (45%), notice period (25%), recruiter response rate (20%), open-to-work flag (10%). |
| **H: Preferred Bonus** | +0.05 max | Nice-to-have bonus signals | LoRA/QLoRA fine-tuning, Learning-to-Rank (LTR/LambdaMART), HR Tech domain experience, distributed inference (Triton/ONNX), high GitHub activity. Each hit adds +0.01 (capped at 0.05). |

### Hard Disqualifiers (Score → 0.0)

| Flag | Logic | Rationale |
|---|---|---|
| `is_honeypot` | Expert proficiency on 5+ skills with 0 months duration, or `grad_year + years_exp > current_year + 2` | Synthetic / spam profiles polluting the dataset |
| `is_consulting_only` | 100% of career history at known consulting firms | JD demands a product-level "0-to-1" builder |
| `is_pure_research` | All titles are research/PhD/postdoc AND no deployment verbs in descriptions | No evidence of production engineering |
| `is_llm_wrapper_only` | < 12 months total AI skill duration AND no pre-LLM ML signals (PyTorch, TF, sklearn) | Recent LLM tourists with no foundational depth |
| `is_not_coding_ic` | Current title contains management keywords AND tenure > 18 months | No longer a hands-on individual contributor |
| `is_wrong_domain` | Primary domain is CV/Speech AND zero NLP/IR keyword hits | Completely wrong specialization |

### Soft Penalizers (Multiplicative)

| Flag | Multiplier | Trigger |
|---|---|---|
| `is_title_chaser` | × 0.75 | >3 jobs with ≤18 months tenure in the last 5 years of career |
| `is_framework_only` | × 0.80 | ≥3 framework-only skills (Langchain, Streamlit…) making up >60% of skill list, with no technique depth keywords |
| `is_closed_source` | × 0.85 | ≥5 years experience but no mentions of papers, talks, GitHub, or OSS |
| Low response rate | × 0.85 | `recruiter_response_rate < 0.10` |
| Notice period | × 0.55–1.0 | Scales by `notice_period_days`: ≤30d=1.0, ≤60d=0.90, ≤90d=0.80, ≤120d=0.70, >120d=0.55 |
| Recency | × 0.70–1.0 | Scales by days since `last_active_date`: ≤90d=1.0, ≤180d=0.85, >180d=0.70 |

---

## Top 10 Results

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

### Score Distribution (Top 100)

| Percentile | Score |
|---|---|
| p50 | 0.6091 |
| p75 | 0.6558 |
| p90 | 0.6972 |
| p95 | 0.7150 |
| p99 | 0.7671 |
| p100 (max) | 0.7808 |

---

## Design Decisions & Tradeoffs

**Why feature rules carry 50% weight, not embeddings:**
Semantic search is powerful but overly forgiving. A candidate who mentions "vectors" in a graphics context can score highly against a JD about "vector databases." By allocating 50% to deterministic features, we guarantee that only candidates with verifiable production retrieval experience, product-company backgrounds, and evaluation depth can reach the top.

**Why consulting-only is a hard disqualifier:**
The JD asks for a high-autonomy "0-to-1" builder who has owned product-level metrics. IT services firms primarily execute outsourced work under client direction. Candidates with 100% consulting careers are structurally unlikely to have the ownership experience this role demands.

**What the pipeline deliberately ignores:**
Education (degrees, universities), certifications, compensation history, and spoken languages. We concluded these are noise for this specific role — a candidate with a top-tier degree but no retrieval experience is less valuable than a self-taught engineer who has shipped FAISS to production.

---

## Known Limitations

- **JD-specific logic**: All keyword lists and feature rules are hardcoded for this particular JD. The system will not generalize to other roles without rewriting feature extraction.
- **Approximate durations**: `start_date` / `end_date` fields are not reliably standardized. Title chaser detection and management tenure calculations are heuristic approximations using `duration_months`.
- **Semantic gaps**: Subtle domain mismatches can slip through if candidates use dense NLP vocabulary to describe unrelated work (e.g., "embedding" in a hardware context).
- **No LLM reasoning**: Reasoning strings are template-generated, not LLM-produced. They are factual and transparent but not conversational.

---

## How to Reproduce

### Option A: Fast Run (Pre-computed Artifacts via Git LFS)

If you have `data/candidates.jsonl`, you can skip the 50–60 minute embedding step because the precomputed artifacts are already in this repo:

```bash
# 1. Install Git LFS and pull the artifacts
git lfs install
git lfs pull

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install sentence-transformers rank_bm25 numpy

# 3. Re-generate only candidates_parsed.pkl (gitignored, 398 MB)
python precompute/01_parse_candidates.py --candidates ../candidates.jsonl  # ~2 min

# 4. Run final ranking
python rank.py --out ../submission.csv                                     # ~1 min

# 5. Validate output
python validate_submission.py
```

All LFS files — `embeddings.npy`, `jd_embedding.npy`, `bm25_scores.pkl`, `feature_matrix.pkl` — are pulled in step 1. Only `candidates_parsed.pkl` needs to be regenerated locally because it is 398 MB and exceeds free-tier LFS quotas.

---

### Option B: Full Pipeline Run (from scratch)

Use this if you want to regenerate every artifact from a raw data file. This single command will run all precompute stages and output the final ranking to the parent directory:

```bash
# 1. Set up environment
python -m venv .venv
source .venv/bin/activate
pip install sentence-transformers rank_bm25 numpy

# 2. Run the pipeline (Stages 1-5 automatically)
python rank.py --candidates ../candidates.jsonl --out ../submission.csv

# 3. Validate output
python validate_submission.py
```

> **Note:** `04_build_embeddings.py` is the bottleneck. It encodes all 100,000 candidates using `all-MiniLM-L6-v2` on CPU. GPU acceleration is supported by `sentence-transformers` if CUDA is available.
