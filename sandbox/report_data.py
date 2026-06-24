import pickle
import csv
import json
import numpy as np

def main():
    print("Loading features...")
    with open('precompute/artifacts/feature_matrix.pkl', 'rb') as f:
        features_list = pickle.load(f)
        
    print("Loading candidates...")
    with open('precompute/artifacts/candidates_parsed.pkl', 'rb') as f:
        candidates = pickle.load(f)
        
    cid_to_cand = {c['candidate_id']: c for c in candidates}
    cid_to_feat = {f['candidate_id']: f for f in features_list}

    # Reconstruct rank.py logic to get distribution of disqualified
    # Actually, rank.py just assigns score=0.0 for disqualified
    
    # 6 Disqualifiers:
    # is_unemployed_too_long
    # is_pure_consulting
    # is_missing_eval_skills
    # is_wrong_domain
    # is_non_technical
    # is_missing_retrieval_tech
    disq_counts = {
        'is_unemployed_too_long': 0,
        'is_pure_consulting': 0,
        'is_missing_eval_skills': 0,
        'is_wrong_domain': 0,
        'is_non_technical': 0,
        'is_missing_retrieval_tech': 0,
    }
    total_disq = 0
    
    # We must calculate scores for ALL candidates exactly as rank.py does
    # to find the true distributions, because team_antigravity.csv only has top 100.
    
    final_scores = []
    
    for f in features_list:
        if f.get('is_unemployed_too_long'):
            disq_counts['is_unemployed_too_long'] += 1
        if f.get('is_pure_consulting'):
            disq_counts['is_pure_consulting'] += 1
        if f.get('is_missing_eval_skills'):
            disq_counts['is_missing_eval_skills'] += 1
        if f.get('is_wrong_domain'):
            disq_counts['is_wrong_domain'] += 1
        if f.get('is_non_technical'):
            disq_counts['is_non_technical'] += 1
        if f.get('is_missing_retrieval_tech'):
            disq_counts['is_missing_retrieval_tech'] += 1
            
        disq = (f.get('is_unemployed_too_long') or 
                f.get('is_pure_consulting') or 
                f.get('is_missing_eval_skills') or 
                f.get('is_wrong_domain') or 
                f.get('is_non_technical') or 
                f.get('is_missing_retrieval_tech'))
                
        if disq:
            total_disq += 1
            final_scores.append(0.0)
            continue
            
        # compute base
        base = (
            0.30 * f.get('production_retrieval_score', 0) +
            0.20 * f.get('product_company_score', 0) +
            0.15 * f.get('eval_depth_score', 0) +
            0.15 * f.get('domain_fit_score', 0) +
            0.10 * f.get('experience_range_score', 0) +
            0.05 * f.get('location_score', 0) +
            0.05 * f.get('engagement_score', 0)
        )
        base = min(base + f.get('preferred_bonus', 0), 1.0)
        multiplier = 1.0
        if f.get('is_title_chaser'): multiplier *= 0.75
        if f.get('is_framework_only'): multiplier *= 0.80
        if f.get('is_closed_source'): multiplier *= 0.85
        if f.get('recruiter_response_rate', 0.5) < 0.10: multiplier *= 0.85
        
        multiplier *= f.get('notice_multiplier', 1.0)
        multiplier *= f.get('recency_multiplier', 1.0)
        
        feature_score = max(0.0, min(1.0, base * multiplier))
        
        # since we don't have bm25/embedding scores for ALL, we'll just parse the CSV for the top 100 distribution
        # Wait, the prompt says: "Report p50, p75, p90, p95, p99, p100 of final_score across all 100 ranked candidates."
        # Not all 100,000! Just the top 100!
        
    print("--- Disqualified Stats ---")
    print(f"Total Disqualified: {total_disq}")
    print(json.dumps(disq_counts, indent=2))
    
    top_100_scores = []
    top_10_details = []
    with open('outputs/team_antigravity.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            top_100_scores.append(float(row['score']))
            
            if int(row['rank']) <= 10:
                cid = row['candidate_id']
                # parse reasoning string to get feature/embed/bm25
                reasoning = row['reasoning']
                # e.g., "feature=0.9419, embedding=0.4170, bm25=0.9235 -> final=0.7808."
                parts = reasoning.split('Final score breakdown: ')[1].split(' ->')[0]
                comps = parts.split(', ')
                f_s = comps[0].split('=')[1]
                e_s = comps[1].split('=')[1]
                b_s = comps[2].split('=')[1]
                
                c_data = cid_to_cand[cid]
                title = c_data.get('career_history', [{}])[0].get('role', 'N/A')
                exp = c_data.get('profile', {}).get('years_of_experience', 'N/A')
                loc = c_data.get('profile', {}).get('location', 'N/A')
                npd = c_data.get('profile', {}).get('notice_period', 'N/A')
                
                top_10_details.append({
                    'Rank': row['rank'],
                    'Candidate ID': cid,
                    'Final Score': row['score'],
                    'Feature Score': f_s,
                    'Embedding Score': e_s,
                    'BM25 Score': b_s,
                    'Title': title,
                    'Experience': f"{exp} yrs",
                    'Location': loc,
                    'Notice Period': f"{npd} days"
                })

    print("\n--- Distribution of Top 100 Ranked Candidates ---")
    print(f"p50: {np.percentile(top_100_scores, 50):.4f}")
    print(f"p75: {np.percentile(top_100_scores, 75):.4f}")
    print(f"p90: {np.percentile(top_100_scores, 90):.4f}")
    print(f"p95: {np.percentile(top_100_scores, 95):.4f}")
    print(f"p99: {np.percentile(top_100_scores, 99):.4f}")
    print(f"p100: {np.max(top_100_scores):.4f}")
    
    print("\n--- Top 10 Candidates Data ---")
    print(json.dumps(top_10_details, indent=2))

if __name__ == '__main__':
    main()
