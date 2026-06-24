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

    # 6 Disqualifiers:
    # is_honeypot
    # is_consulting_only
    # is_pure_research
    # is_llm_wrapper_only
    # is_not_coding_ic
    # is_wrong_domain
    disq_counts = {
        'is_honeypot': 0,
        'is_consulting_only': 0,
        'is_pure_research': 0,
        'is_llm_wrapper_only': 0,
        'is_not_coding_ic': 0,
        'is_wrong_domain': 0,
    }
    total_disq = 0
    
    for f in features_list:
        if f.get('is_honeypot'): disq_counts['is_honeypot'] += 1
        if f.get('is_consulting_only'): disq_counts['is_consulting_only'] += 1
        if f.get('is_pure_research'): disq_counts['is_pure_research'] += 1
        if f.get('is_llm_wrapper_only'): disq_counts['is_llm_wrapper_only'] += 1
        if f.get('is_not_coding_ic'): disq_counts['is_not_coding_ic'] += 1
        if f.get('is_wrong_domain'): disq_counts['is_wrong_domain'] += 1
            
        disq = (f.get('is_honeypot') or 
                f.get('is_consulting_only') or 
                f.get('is_pure_research') or 
                f.get('is_llm_wrapper_only') or 
                f.get('is_not_coding_ic') or 
                f.get('is_wrong_domain'))
                
        if disq:
            total_disq += 1

    print("--- Disqualified Stats ---")
    print(f"Total Disqualified: {total_disq}")
    print(json.dumps(disq_counts, indent=2))
    
if __name__ == '__main__':
    main()
