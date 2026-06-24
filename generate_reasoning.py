def generate(rank, candidate, features, feature_score, emb_score, bm25_score, final_score):
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    
    # 1. Experience
    years_exp = profile.get('years_of_experience', 0)
    current_title = profile.get('current_title', 'Unknown Title')
    if not current_title: current_title = 'Unknown Title'
    
    current_company = career[0].get('company', 'Unknown Company') if career else 'Unknown Company'
    if not current_company: current_company = 'Unknown Company'
        
    location = profile.get('location', 'Unknown Location')
    if not location: location = 'Unknown Location'
    
    notice_period = signals.get('notice_period_days', 0)
    if notice_period is None: notice_period = 0
    
    part1 = f"Candidate has {years_exp} years of experience, currently working as {current_title} at {current_company} in {location} with a notice period of {notice_period} days."
    
    # 2. Retrieval tech
    RETRIEVAL_TECH = ['pinecone','qdrant','weaviate','milvus','faiss','elasticsearch','opensearch','vector database','vector search','ann']
    
    skills_list = [s.get('name', '').lower() for s in skills if s.get('name')]
    skills_found = list(set([t for t in RETRIEVAL_TECH for s in skills_list if t in s]))
    
    work_texts = [c.get('description', '').lower() for c in career if c.get('description')]
    work_text_combined = " ".join(work_texts)
    prod_found = list(set([t for t in RETRIEVAL_TECH if t in work_text_combined]))
    
    if prod_found and skills_found:
        part2 = f"Retrieval tech keywords found in production context ({', '.join(prod_found)}) and in skills list ({', '.join(skills_found)})."
    elif prod_found:
        part2 = f"Retrieval tech keywords found in production context only ({', '.join(prod_found)})."
    elif skills_found:
        part2 = f"Retrieval tech keywords found in skills list only ({', '.join(skills_found)})."
    else:
        part2 = "No retrieval tech keywords were found in either skills or production context."
        
    # 3. Eval keywords
    EVAL_KEYWORDS = ['ndcg','mrr','map','mean average precision','a/b test',
                     'offline evaluation','online evaluation','offline-to-online',
                     'ranking evaluation','precision@k','recall@k','click-through']
    
    summary_text = profile.get('summary', '').lower() if profile.get('summary') else ''
    profile_text = summary_text + " " + work_text_combined
    
    eval_found = list(set([k for k in EVAL_KEYWORDS if k in profile_text]))
    if eval_found:
        part3 = f"Evaluation keywords found: {', '.join(eval_found)}."
    else:
        part3 = "No evaluation keywords were found."
        
    # 4. Product company fraction
    CONSULTING_FIRMS_SET = {'tcs','infosys','wipro','accenture','cognizant','capgemini',
                            'hcl','tech mahindra','mphasis','hexaware','l&t infotech'}
    
    total_months = 0
    prod_months = 0
    prod_companies = []
    
    for c in career:
        c_name = c.get('company', '')
        if not c_name: c_name = 'Unknown'
        c_name_lower = c_name.lower()
        c_type = c.get('industry', '')
        if not c_type: c_type = ''
        c_type = c_type.lower()
        dur = c.get('duration_months', 0)
        if dur is None: dur = 0
        
        total_months += dur
        if c_name_lower not in CONSULTING_FIRMS_SET and c_type not in ['academic', 'lab', 'research_org']:
            prod_months += dur
            if c_name not in prod_companies:
                prod_companies.append(c_name)
                
    fraction = (prod_months / total_months * 100) if total_months > 0 else 0
    comp_str = ", ".join(prod_companies) if prod_companies else "None"
    part4 = f"Product company fraction is {fraction:.0f}% of their career (companies: {comp_str})."
    
    # 5. Soft penalizers
    penalizers = []
    if features.get('is_title_chaser'): penalizers.append("is_title_chaser")
    if features.get('is_framework_only'): penalizers.append("is_framework_only")
    if features.get('is_closed_source'): penalizers.append("is_closed_source")
    
    notice_mult = features.get('notice_multiplier', 1.0)
    if notice_mult < 1.0: penalizers.append(f"notice_multiplier={notice_mult}")
    
    rec_mult = features.get('recency_multiplier', 1.0)
    if rec_mult < 1.0: penalizers.append(f"recency_multiplier={rec_mult}")
    
    if penalizers:
        part5 = f"Soft penalizers fired: {', '.join(penalizers)}."
    else:
        part5 = "No soft penalizers fired."
        
    # 6. Score breakdown
    part6 = f"Final score breakdown: feature={feature_score:.4f}, embedding={emb_score:.4f}, bm25={bm25_score:.4f} -> final={final_score:.4f}."
    
    return " ".join([part1, part2, part3, part4, part5, part6])
