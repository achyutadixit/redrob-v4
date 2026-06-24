import json
import pickle
import datetime

def compute_features(cand):
    # features will store all inputs to the ranking score
    features = {}
    features['candidate_id'] = cand.get('candidate_id')
    features['is_honeypot'] = cand.get('is_honeypot', False)
    
    profile = cand.get('profile', {})
    career = cand.get('career_history', [])
    skills = cand.get('skills', [])
    signals = cand.get('redrob_signals', {})
    
    work_history_text = " ".join([c.get('description', '').lower() for c in career if c.get('description')])
    work_titles_text = " ".join([c.get('title', '').lower() for c in career if c.get('title')])
    summary_text = profile.get('summary', '')
    if not summary_text: summary_text = ''
    headline = profile.get('headline', '')
    if not headline: headline = ''
    summary_text = summary_text.lower()
    headline = headline.lower()
    profile_text = summary_text + " " + headline + " " + work_history_text + " " + work_titles_text
    
    # 1.1 Hard Disqualifiers
    CONSULTING_FIRMS = {'tcs','infosys','wipro','accenture','cognizant','capgemini',
                        'hcl','tech mahindra','mphasis','hexaware','l&t infotech'}
    if not career:
        features['is_consulting_only'] = False
    else:
        features['is_consulting_only'] = all(c.get('company', '').lower() in CONSULTING_FIRMS for c in career)
        
    research_titles = ['researcher', 'research scientist', 'phd', 'postdoc']
    is_pure_research_titles = False
    if career:
        is_pure_research_titles = all(any(rt in c.get('title', '').lower() for rt in research_titles) for c in career)
    has_deployed = any(w in work_history_text for w in ['deployed', 'production', 'shipped'])
    features['is_pure_research'] = is_pure_research_titles and not has_deployed
    
    ai_skills = ['ai', 'machine learning', 'ml', 'deep learning', 'nlp', 'llm', 'generative ai']
    ai_duration = sum(s.get('duration_months', 0) for s in skills if s.get('name', '').lower() in ai_skills)
    pre_llm_signals = ['tf', 'tensorflow', 'pytorch', 'sklearn', 'embedding']
    has_pre_llm = any(s in profile_text for s in pre_llm_signals)
    features['is_llm_wrapper_only'] = (ai_duration < 12) and not has_pre_llm
    
    current_title = profile.get('current_title', '')
    if not current_title: current_title = ''
    current_title = current_title.lower()
    ic_flags = ['architect', 'vp', 'cto', 'director', 'head of']
    has_mgmt_title = any(f in current_title for f in ic_flags)
    mgmt_duration = 0
    if has_mgmt_title and career:
        if career[0].get('is_current'):
            mgmt_duration = career[0].get('duration_months', 0)
    features['is_not_coding_ic'] = has_mgmt_title and mgmt_duration > 18
    
    cv_keywords = ['computer vision', 'speech', 'object detection', 'image', 'cnn', 'yolo', 'segmentation', 'asr', 'tts', 'audio']
    primary_domain_cv = any(k in profile.get('summary', '').lower() for k in cv_keywords) if profile.get('summary') else False
    nlp_keywords = ['nlp', 'ir', 'retrieval']
    has_nlp = any(k in profile_text for k in nlp_keywords)
    features['is_wrong_domain'] = primary_domain_cv and not has_nlp
    
    # 1.2 Soft Penalizers
    recent_jobs = []
    total_months = 0
    for job in career:
        if total_months >= 60:
            break
        recent_jobs.append(job)
        total_months += job.get('duration_months', 0)
        
    short_recent = sum(1 for j in recent_jobs if j.get('duration_months', 99) <= 18)
    features['is_title_chaser'] = short_recent > 3
    
    FRAMEWORK_NAMES = ['langchain','streamlit','gradio','fastapi','flask','django',
                       'huggingface','openai','llamaindex','llamacpp','ollama',
                       'langsmith','flowise','chainlit','autogen','crewai']
    TECHNIQUE_DEPTH = ['attention','transformer architecture','fine-tun','lora','qlora',
                       'embedding','vector','retrieval','bm25','ndcg','mrr',
                       'loss function','gradient','backprop','training loop',
                       'hyperparameter','regularization','cross-entropy','cosine similarity']

    skill_names = [s.get('name', '').lower() for s in skills if s.get('name')]
    framework_count = sum(1 for s in skill_names if any(f in s for f in FRAMEWORK_NAMES))
    technique_present = any(t in profile_text for t in TECHNIQUE_DEPTH)

    features['is_framework_only'] = (
        framework_count >= 3
        and len(skill_names) > 0
        and (framework_count / len(skill_names)) > 0.6
        and not technique_present
    )
    
    years_exp = profile.get('years_of_experience', 0)
    if years_exp is None: years_exp = 0
    has_oss = any(k in profile_text for k in ['paper', 'talk', 'github', 'oss', 'open source'])
    features['is_closed_source'] = years_exp >= 5 and not has_oss
    
    np_days = signals.get('notice_period_days', 0)
    if np_days is None: np_days = 0
    if np_days <= 30: notice_mult = 1.0
    elif np_days <= 60: notice_mult = 0.90
    elif np_days <= 90: notice_mult = 0.80
    elif np_days <= 120: notice_mult = 0.70
    else: notice_mult = 0.55
    features['notice_multiplier'] = notice_mult
    
    last_active = signals.get('last_active_date', '2024-01-01')
    try:
        last_active_dt = datetime.datetime.strptime(last_active, '%Y-%m-%d').date()
    except:
        last_active_dt = datetime.date.today() - datetime.timedelta(days=200)
    days_since_active = (datetime.date.today() - last_active_dt).days
    
    if days_since_active <= 90: recency_mult = 1.0
    elif days_since_active <= 180: recency_mult = 0.85
    else: recency_mult = 0.70
    features['recency_multiplier'] = recency_mult
    
    # Feature Groups
    RETRIEVAL_TECH = ['pinecone','qdrant','weaviate','milvus','faiss','elasticsearch',
                      'opensearch','vector database','vector search','ann']
    RETRIEVAL_TECHNIQUE = ['hybrid retrieval','hybrid search','dense retrieval','bm25',
                           'sentence-transformers','bge','e5','embedding drift',
                           'index refresh','retrieval regression','semantic search']
    SYSTEM_TYPES = ['recommendation system','ranking system','search system',
                    'information retrieval','candidate ranking','document retrieval']
    PRODUCTION_VERBS = ['built','deployed','shipped','owned','led','designed',
                        'implemented','serving','production','scaled']
                        
    has_vector_db_production = False
    for c in career:
        desc = c.get('description', '')
        if not desc: desc = ''
        desc = desc.lower()
        if any(t in desc for t in RETRIEVAL_TECH) and any(v in desc for v in PRODUCTION_VERBS):
            has_vector_db_production = True
            break
            
    has_embedding_retrieval = any(t in work_history_text for t in RETRIEVAL_TECHNIQUE)
    has_rec_or_rank_system = any(t in work_history_text for t in SYSTEM_TYPES)
    has_hybrid_search = 'hybrid' in work_history_text
    has_operational_signals = any(w in work_history_text for w in ['drift', 'refresh', 'regression', 'latency'])
    retrieval_tech_count = min(sum(1 for t in RETRIEVAL_TECH if t in profile_text) / 10.0, 1.0)
    
    features['production_retrieval_score'] = (
        0.30 * float(has_vector_db_production) +
        0.25 * float(has_embedding_retrieval)  +
        0.20 * float(has_rec_or_rank_system)   +
        0.10 * float(has_hybrid_search)        +
        0.10 * float(has_operational_signals)  +
        0.05 * retrieval_tech_count
    )
    
    EVAL_KEYWORDS = ['ndcg','mrr','map','mean average precision','a/b test',
                     'offline evaluation','online evaluation','offline-to-online',
                     'ranking evaluation','precision@k','recall@k','click-through']
    
    eval_matches = sum(1 for k in EVAL_KEYWORDS if k in profile_text)
    has_strong_eval = ('designed evaluation' in profile_text) or ('built eval framework' in profile_text)
    if eval_matches == 0 and not has_strong_eval:
        eval_score = 0.0
    elif eval_matches in [1, 2] and not has_strong_eval:
        eval_score = 0.5
    else:
        eval_score = 1.0
    features['eval_depth_score'] = eval_score
    
    CONSULTING_FIRMS_SET = {'tcs','infosys','wipro','accenture','cognizant','capgemini',
                            'hcl','tech mahindra','mphasis','hexaware','l&t infotech'}
    product_company_years = 0
    total_years = 0
    for c in career:
        c_name = c.get('company', '')
        if not c_name: c_name = ''
        c_name = c_name.lower()
        c_type = c.get('industry', '')
        if not c_type: c_type = ''
        c_type = c_type.lower()
        dur = c.get('duration_months', 0) / 12.0
        total_years += dur
        if c_name not in CONSULTING_FIRMS_SET and c_type not in ['academic', 'lab', 'research_org']:
            product_company_years += dur
            
    product_company_fraction = product_company_years / max(total_years, 1.0)
    has_shipped_to_users = any(w in work_history_text for w in ['users', 'customers', 'production'])
    features['product_company_score'] = (
        0.50 * product_company_fraction +
        0.30 * float(has_shipped_to_users) +
        0.20 * min(product_company_years / 5.0, 1.0)
    )
    
    if years_exp < 3: exp_score = 0.20
    elif years_exp < 5: exp_score = 0.50
    elif years_exp <= 9: exp_score = 1.00
    elif years_exp <= 12: exp_score = 0.80
    else: exp_score = 0.60
    features['experience_range_score'] = exp_score
    
    NLP_IR_KEYWORDS = ['nlp','natural language','text','retrieval','search','embedding',
                       'ranking','rag','bert','transformer','information retrieval','ir']
    CV_KEYWORDS     = ['computer vision','object detection','image','cnn','yolo','segmentation']
    SPEECH_KEYWORDS = ['speech','asr','tts','audio','wav2vec']
    
    nlp_score = sum(1 for k in NLP_IR_KEYWORDS if k in profile_text)
    cv_score = sum(1 for k in CV_KEYWORDS if k in profile_text)
    speech_score = sum(1 for k in SPEECH_KEYWORDS if k in profile_text)
    
    if nlp_score >= cv_score and nlp_score >= speech_score and nlp_score > 0:
        dom_score = 1.00
    elif nlp_score > 0:
        dom_score = 0.80
    elif cv_score == 0 and speech_score == 0:
        dom_score = 0.50
    else:
        dom_score = 0.20
    features['domain_fit_score'] = dom_score
    
    PREFERRED   = {'pune', 'noida'}
    ACCEPTABLE  = {'hyderabad', 'mumbai', 'delhi', 'delhi ncr', 'gurgaon', 'gurugram'}
    TIER1_OTHER = {'bangalore', 'bengaluru', 'chennai', 'kolkata'}
    
    city = profile.get('location', '')
    if not city: city = ''
    city = city.lower().strip()
    open_to_relocation = signals.get('willing_to_relocate', False)
    
    if city in PREFERRED or open_to_relocation:
        loc_score = 1.00
    elif city in ACCEPTABLE:
        loc_score = 0.90
    elif city in TIER1_OTHER:
        loc_score = 0.80
    else:
        loc_score = 0.50
    features['location_score'] = loc_score
    
    if days_since_active <= 30: rec_score = 1.00
    elif days_since_active <= 90: rec_score = 0.85
    elif days_since_active <= 180: rec_score = 0.65
    else: rec_score = 0.40
    
    if np_days <= 30: not_score = 1.00
    elif np_days <= 60: not_score = 0.85
    elif np_days <= 90: not_score = 0.70
    elif np_days <= 120: not_score = 0.55
    else: not_score = 0.35
    
    response_rate = signals.get('recruiter_response_rate', 0.5)
    if response_rate is None or response_rate < 0: response_rate = 0.5
    features['recruiter_response_rate'] = response_rate
    
    open_to_work = signals.get('open_to_work_flag', False)
    features['engagement_score'] = (
        0.45 * rec_score +
        0.25 * not_score +
        0.20 * response_rate +
        0.10 * float(open_to_work)
    )
    
    github_score = signals.get('github_activity_score', -1)
    if github_score is None: github_score = -1
    github_signal = (github_score > 30) or any(k in profile_text for k in ['open source','open-source','oss contribution'])

    preferred_hits = sum([
        any(k in profile_text for k in ['lora','qlora','peft','fine-tun']),
        any(k in profile_text for k in ['learning to rank','ltr','xgboost rank','lambdamart']),
        any(k in profile_text for k in ['hr tech','hrtech','recruiting','talent','ats']),
        any(k in profile_text for k in ['distributed','inference optimization','triton','onnx']),
        github_signal,
    ])
    features['preferred_bonus'] = preferred_hits * 0.01
    
    return features

def main():
    import pickle
    with open('precompute/artifacts/candidates_parsed.pkl', 'rb') as f:
        candidates = pickle.load(f)
        
    features_list = []
    for cand in candidates:
        features_list.append(compute_features(cand))
        
    with open('precompute/artifacts/feature_matrix.pkl', 'wb') as f:
        pickle.dump(features_list, f)
        
    print(f"Extracted features for {len(features_list)} candidates.")

if __name__ == '__main__':
    main()
