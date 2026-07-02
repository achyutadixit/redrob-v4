import json
import pickle
import datetime
import os

def check_honeypot(cand):
    reasons = []

    # Check 1: Expert proficiency on 5+ skills with 0 months duration.
    #
    # This is the primary signal described in the hackathon spec: synthetic profiles
    # that claim expert-level mastery of many skills without any duration. Real
    # candidates occasionally have one or two skills with 0 months (e.g. freshly
    # adopted tools), but 5+ is a strong synthetic-profile signal.
    #
    # The temporal check (grad_year + years_exp > current_year) was removed because:
    # 1. The dataset has no start_year fields in career_history, so the only cross-
    #    check available (career start year) never fires.
    # 2. The education end_year is the highest degree end year, which can be recent
    #    even for experienced engineers who pursued postgrad while working.
    # 3. All temporally-suspicious profiles (Accountants, Marketing Managers, etc.)
    #    score near zero naturally via feature scoring — per the hackathon spec,
    #    "a good ranking system will naturally avoid them."
    skills = cand.get('skills', [])
    expert_zero_years = sum(
        1 for s in skills
        if s.get('proficiency', '').lower() == 'expert'
        and (s.get('duration_months') or 0) == 0
    )
    if expert_zero_years >= 5:
        reasons.append(f"expert proficiency on {expert_zero_years} skills with 0 months duration")

    return len(reasons) > 0, reasons


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=str, default='../candidates.jsonl')
    args = parser.parse_args()
    data_path = args.candidates

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
