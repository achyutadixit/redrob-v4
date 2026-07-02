import pickle
import numpy as np
import csv
import generate_reasoning
import os
import sys
import argparse
import subprocess

ARTIFACT_DIR = 'precompute/artifacts'
ARTIFACTS_NEEDED = [
    'embeddings.npy',
    'jd_embedding.npy',
    'bm25_scores.pkl',
    'feature_matrix.pkl',
]

def compute_score(features):
    # Hard disqualifiers
    if features.get('is_honeypot') or features.get('is_consulting_only') or \
       features.get('is_pure_research') or features.get('is_llm_wrapper_only') or \
       features.get('is_not_coding_ic') or features.get('is_wrong_domain'):
        return 0.0
        
    base = (
        0.30 * features.get('production_retrieval_score', 0) +
        0.20 * features.get('product_company_score', 0) +
        0.15 * features.get('eval_depth_score', 0) +
        0.15 * features.get('domain_fit_score', 0) +
        0.10 * features.get('experience_range_score', 0) +
        0.05 * features.get('location_score', 0) +
        0.05 * features.get('engagement_score', 0)
    )
    
    base = min(base + features.get('preferred_bonus', 0), 1.0)
    
    multiplier = 1.0
    if features.get('is_title_chaser'): multiplier *= 0.75
    if features.get('is_framework_only'): multiplier *= 0.80
    if features.get('is_closed_source'): multiplier *= 0.85
    if features.get('recruiter_response_rate', 0.5) < 0.10: multiplier *= 0.85
    
    multiplier *= features.get('notice_multiplier', 1.0)
    multiplier *= features.get('recency_multiplier', 1.0)
    
    return max(0.0, min(1.0, base * multiplier))


def download_artifacts_from_hf(hf_repo):
    """Download missing precomputed artifacts from a public Hugging Face Hub dataset repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    print(f"Downloading artifacts from Hugging Face Hub: {hf_repo}")
    for filename in ARTIFACTS_NEEDED:
        dest = os.path.join(ARTIFACT_DIR, filename)
        if os.path.exists(dest):
            print(f"  [SKIP] {filename} already exists locally.")
            continue
        print(f"  Downloading {filename}...")
        local_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            repo_type="dataset",
            local_dir=ARTIFACT_DIR,
        )
        print(f"  Saved to {local_path}")


def artifacts_exist():
    return all(
        os.path.exists(os.path.join(ARTIFACT_DIR, f))
        for f in ARTIFACTS_NEEDED
    )


def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Hackathon")
    parser.add_argument('--candidates', type=str,
                        help='Path to candidates.jsonl. If provided, runs the full precompute pipeline.')
    parser.add_argument('--out', type=str, default='../submission.csv',
                        help='Path to output CSV (default: ../submission.csv)')
    parser.add_argument('--hf-repo', type=str, default=None,
                        help='Hugging Face Hub dataset repo id to download artifacts from, '
                             'e.g. achyutadixit/redrob-v4-artifacts. '
                             'Artifacts are only downloaded if not already present locally.')
    args = parser.parse_args()

    # --- Step 1: Resolve artifacts ---
    if args.candidates:
        # Full pipeline: regenerate every artifact from raw JSONL
        print(f"Running full precompute pipeline for candidates: {args.candidates}")
        python_exec = sys.executable
        subprocess.run([python_exec, "precompute/01_parse_candidates.py",
                        "--candidates", args.candidates], check=True)
        subprocess.run([python_exec, "precompute/02_extract_features.py"], check=True)
        subprocess.run([python_exec, "precompute/03_build_bm25.py"], check=True)
        subprocess.run([python_exec, "precompute/04_build_embeddings.py"], check=True)
    else:
        # Fast path: use pre-computed artifacts
        if args.hf_repo and not artifacts_exist():
            # Auto-download from HF Hub if requested and artifacts are missing
            download_artifacts_from_hf(args.hf_repo)
        elif not artifacts_exist():
            print("[ERROR] Precomputed artifacts not found.")
            print("  Option A (recommended): git lfs pull")
            print("  Option B: python rank.py --hf-repo achyutadixit/redrob-v4-artifacts --candidates ../candidates.jsonl")
            print("  Option C: python rank.py --candidates ../candidates.jsonl  (full pipeline, ~60 min)")
            sys.exit(1)

        # candidates_parsed.pkl must still be generated from the raw JSONL if missing
        parsed_pkl = os.path.join(ARTIFACT_DIR, 'candidates_parsed.pkl')
        if not os.path.exists(parsed_pkl):
            candidates_path = args.candidates or '../candidates.jsonl'
            if not os.path.exists(candidates_path):
                print(f"[ERROR] candidates_parsed.pkl not found and no candidates.jsonl available.")
                print(f"  Please provide --candidates path or place candidates.jsonl at: {candidates_path}")
                sys.exit(1)
            print(f"Parsing candidates from {candidates_path} (stage 1 only, ~2 min)...")
            subprocess.run([sys.executable, "precompute/01_parse_candidates.py",
                            "--candidates", candidates_path], check=True)

    # --- Step 2: Load artifacts ---
    print("Loading artifacts...")
    with open(os.path.join(ARTIFACT_DIR, 'candidates_parsed.pkl'), 'rb') as f:
        candidates = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, 'feature_matrix.pkl'), 'rb') as f:
        features_list = pickle.load(f)
    with open(os.path.join(ARTIFACT_DIR, 'bm25_scores.pkl'), 'rb') as f:
        bm25_dict = pickle.load(f)
        
    embeddings = np.load(os.path.join(ARTIFACT_DIR, 'embeddings.npy'))
    jd_embedding = np.load(os.path.join(ARTIFACT_DIR, 'jd_embedding.npy'))
    
    # --- Step 3: Score ---
    print("Computing embedding cosines...")
    jd_norm = jd_embedding / np.linalg.norm(jd_embedding)
    emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norms[emb_norms == 0] = 1.0
    normalized_embs = embeddings / emb_norms
    cosines = np.dot(normalized_embs, jd_norm)
    
    print("Ranking candidates...")
    results = []
    for idx, cand in enumerate(candidates):
        cid = cand.get('candidate_id')
        features = features_list[idx]
        feature_score = compute_score(features)
        
        emb_score = max(0.0, float(cosines[idx]))
        bm25_score = bm25_dict.get(cid, 0.0)
            
        if feature_score == 0.0:
            final_score = 0.0
        else:
            final_score = (
                0.50 * feature_score +
                0.30 * emb_score +
                0.20 * bm25_score
            )
            
        results.append((final_score, cid, cand, idx, features, feature_score, emb_score, bm25_score))
        
    results.sort(key=lambda x: (-x[0], x[1]))
    
    print("Generating reasoning for top candidates...")
    
    output_rows = []
    top_n = min(100, len(results))
    for rank in range(1, top_n + 1):
        score, cid, cand, orig_idx, features, feature_score, emb_score, bm25_score = results[rank - 1]
        
        reasoning = generate_reasoning.generate(rank, cand, features, feature_score, emb_score, bm25_score, score)
            
        output_rows.append({
            'rank': rank,
            'candidate_id': cid,
            'score': round(score, 4),
            'reasoning': reasoning
        })
        
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['rank', 'candidate_id', 'score', 'reasoning'])
        writer.writeheader()
        writer.writerows(output_rows)
        
    print(f"Done. Saved to {args.out}")

if __name__ == '__main__':
    main()
