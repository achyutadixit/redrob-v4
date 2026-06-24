import pickle
import csv
import json

def main():
    print("Loading features...")
    with open('precompute/artifacts/feature_matrix.pkl', 'rb') as f:
        features_list = pickle.load(f)
        
    print("Loading candidates...")
    with open('precompute/artifacts/candidates_parsed.pkl', 'rb') as f:
        candidates = pickle.load(f)

    # Let's see how many affected globally
    fw_only_count = 0
    rr_penalizer_count = 0
    title_chaser_count = 0
    
    cid_to_features = {f['candidate_id']: f for f in features_list}
    cid_to_cand = {c['candidate_id']: c for c in candidates}

    for f in features_list:
        if f.get('is_framework_only'): fw_only_count += 1
        if f.get('recruiter_response_rate', 0.5) < 0.10: rr_penalizer_count += 1
        if f.get('is_title_chaser'): title_chaser_count += 1
        
    print("\n--- GLOBAL STATS ---")
    print(f"Candidates affected by Fix 1 (is_framework_only): {fw_only_count}")
    print(f"Candidates affected by Fix 2 (recruiter_response_rate < 0.10): {rr_penalizer_count}")
    print(f"Candidates affected by Fix 3 (is_title_chaser): {title_chaser_count}")
    
    # Load rank output
    results = []
    with open('outputs/team_antigravity.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
            
    print("\n--- NEW TOP 20 ---")
    print("Rank | Candidate ID | Final Score | Feature Score | Fixes Fired")
    
    for row in results[:20]:
        rank = row['rank']
        cid = row['candidate_id']
        final_score = row['score']
        f = cid_to_features.get(cid, {})
        cand = cid_to_cand.get(cid, {})
        
        # We need the feature_score...
        # Wait, the feature score is computed in rank.py and not saved in feature_dict.
        # Let's compute it.
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
        
        fixes = []
        if f.get('is_framework_only'): fixes.append("Fix 1")
        if f.get('recruiter_response_rate', 0.5) < 0.10: fixes.append("Fix 2")
        if f.get('is_title_chaser'): fixes.append("Fix 3")
        
        # Check fix 4: consulting firm l&t infotech
        career = cand.get('career_history', [])
        has_lt = any(c.get('company', '').lower() == 'l&t infotech' for c in career)
        if has_lt and all(c.get('company', '').lower() in {'tcs','infosys','wipro','accenture','cognizant','capgemini','hcl','tech mahindra','mphasis','hexaware','l&t infotech'} for c in career):
            fixes.append("Fix 4")
            
        # Fix 5: headline added
        headline = cand.get('profile', {}).get('headline', '')
        if headline: fixes.append("Fix 5")
        
        # Fix 6: github score > 30
        github_score = cand.get('redrob_signals', {}).get('github_activity_score', -1)
        if github_score is not None and github_score > 30: fixes.append("Fix 6")
        
        # Fix 7: open to work
        otw = cand.get('redrob_signals', {}).get('open_to_work_flag', False)
        if otw: fixes.append("Fix 7")
        
        fixes_str = ", ".join(fixes) if fixes else "None"
        
        print(f"{rank:4} | {cid:12} | {final_score:11} | {feature_score:13.4f} | {fixes_str}")

if __name__ == '__main__':
    main()
