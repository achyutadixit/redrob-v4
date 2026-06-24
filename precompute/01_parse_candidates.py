import json
import pickle
import datetime
import os

def check_honeypot(cand):
    reasons = []
    
    # Check 1: expert proficiency on 5+ skills with years_used=0
    skills = cand.get('skills', [])
    expert_zero_years = 0
    for s in skills:
        prof = s.get('proficiency', '').lower()
        duration = s.get('duration_months', 0)
        if prof == 'expert' and duration == 0:
            expert_zero_years += 1
            
    if expert_zero_years >= 5:
        reasons.append("expert proficiency on 5+ skills with years_used=0")
        
    # Check 2: grad_year + years_exp > current_year + 2
    education = cand.get('education', [])
    grad_year = None
    for edu in education:
        ey = edu.get('end_year')
        if ey:
            if grad_year is None or ey > grad_year:
                grad_year = ey
                
    years_exp = cand.get('profile', {}).get('years_of_experience', 0)
    current_year = datetime.datetime.now().year
    
    if grad_year is not None and years_exp is not None:
        if grad_year + years_exp > current_year + 2:
            reasons.append("grad_year + years_exp > current_year + 2")
            
    return len(reasons) > 0, reasons

def main():
    data_path = 'data/candidates.jsonl'
    
    candidates = []
    with open(data_path, 'r') as f:
        for line in f:
            candidates.append(json.loads(line.strip()))
            
    parsed = []
    honeypots = {}
    
    for cand in candidates:
        is_honey, reasons = check_honeypot(cand)
        cand['is_honeypot'] = is_honey
        if is_honey:
            honeypots[cand['candidate_id']] = reasons
        parsed.append(cand)
        
    os.makedirs('precompute/artifacts', exist_ok=True)
    
    with open('precompute/artifacts/candidates_parsed.pkl', 'wb') as f:
        pickle.dump(parsed, f)
        
    with open('precompute/artifacts/honeypots_flagged.json', 'w') as f:
        json.dump(honeypots, f, indent=2)
        
    print(f"Parsed {len(parsed)} candidates. Found {len(honeypots)} honeypots.")

if __name__ == '__main__':
    main()
