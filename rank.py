import pickle
import numpy as np
import csv
import generate_reasoning
import os
import argparse
import subprocess

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

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob Hackathon")
    parser.add_argument('--candidates', type=str, help='Path to candidates.jsonl. If provided, runs the full precompute pipeline.', default='../candidates.jsonl')
    parser.add_argument('--out', type=str, default='../submission.csv', help='Path to output CSV')
    args = parser.parse_args()

    if args.candidates:
        print(f"Running full precompute pipeline for candidates: {args.candidates}")
        subprocess.run(["python", "precompute/01_parse_candidates.py", "--candidates", args.candidates], check=True)
        subprocess.run(["python", "precompute/02_extract_features.py"], check=True)
        subprocess.run(["python", "precompute/03_build_bm25.py"], check=True)
        subprocess.run(["python", "precompute/04_build_embeddings.py"], check=True)

    print("Loading artifacts...")
    with open('precompute/artifacts/candidates_parsed.pkl', 'rb') as f:
        candidates = pickle.load(f)
    with open('precompute/artifacts/feature_matrix.pkl', 'rb') as f:
        features_list = pickle.load(f)
    with open('precompute/artifacts/bm25_scores.pkl', 'rb') as f:
        bm25_dict = pickle.load(f)
        
    embeddings = np.load('precompute/artifacts/embeddings.npy')
    jd_embedding = np.load('precompute/artifacts/jd_embedding.npy')
    
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
