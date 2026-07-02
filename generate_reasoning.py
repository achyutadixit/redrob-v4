import random

def generate(rank, candidate, features, feature_score, emb_score, bm25_score, final_score):
    cid = candidate.get('candidate_id', 'Unknown')
    rng = random.Random(cid)
    
    profile = candidate.get('profile', {})
    signals = candidate.get('redrob_signals', {})
    skills = candidate.get('skills', [])
    career = candidate.get('career_history', [])
    
    # --- Facts Extraction ---
    years_exp = profile.get('years_of_experience', 0)
    current_title = profile.get('current_title', '') or 'Software Engineer'
    current_company = career[0].get('company', '') if career else ''
    if not current_company:
        current_company = 'their current organization'
        
    notice_period = signals.get('notice_period_days', 0) or 0
    recruiter_response = signals.get('recruiter_response_rate', 0.5)
    if recruiter_response is None: recruiter_response = 0.5
    
    RETRIEVAL_TECH = ['pinecone','qdrant','weaviate','milvus','faiss','elasticsearch','opensearch','vector database','vector search','ann']
    skills_list = [s.get('name', '').lower() for s in skills if s.get('name')]
    skills_found = list(set([t for t in RETRIEVAL_TECH for s in skills_list if t in s]))
    
    work_texts = [c.get('description', '').lower() for c in career if c.get('description')]
    work_text_combined = " ".join(work_texts)
    prod_found = list(set([t for t in RETRIEVAL_TECH if t in work_text_combined]))
    
    EVAL_KEYWORDS = ['ndcg','mrr','map','mean average precision','a/b test',
                     'offline evaluation','online evaluation','offline-to-online',
                     'ranking evaluation','precision@k','recall@k','click-through']
    summary_text = profile.get('summary', '').lower() if profile.get('summary') else ''
    profile_text = summary_text + " " + work_text_combined
    eval_found = list(set([k for k in EVAL_KEYWORDS if k in profile_text]))
    
    CONSULTING_FIRMS_SET = {'tcs','infosys','wipro','accenture','cognizant','capgemini',
                            'hcl','tech mahindra','mphasis','hexaware','l&t infotech'}
    prod_months = 0
    total_months = 0
    for c in career:
        c_name_lower = (c.get('company', '') or '').lower()
        c_type = (c.get('industry', '') or '').lower()
        dur = c.get('duration_months', 0) or 0
        total_months += dur
        if c_name_lower not in CONSULTING_FIRMS_SET and c_type not in ['academic', 'lab', 'research_org']:
            prod_months += dur
    fraction = (prod_months / total_months * 100) if total_months > 0 else 0

    if rank <= 15:
        tone = "glowing"
    elif rank <= 50:
        tone = "solid"
    else:
        tone = "marginal"

    sentences = []

    intro_opts_glowing = [
        f"This candidate is an exceptionally strong fit, bringing {years_exp} years of experience from {current_company}.",
        f"Ranking in the top tier, this {current_title} at {current_company} demonstrates deep alignment with our core requirements.",
        f"A standout profile with {years_exp} years of industry experience, currently serving as a {current_title} at {current_company}."
    ]
    intro_opts_solid = [
        f"A solid candidate with {years_exp} years of experience, presently at {current_company}.",
        f"Demonstrates good potential as a {current_title} with {years_exp} years of relevant background, currently at {current_company}.",
        f"This candidate has a respectable track record spanning {years_exp} years, working most recently at {current_company}."
    ]
    intro_opts_marginal = [
        f"While they made the top 100, this {current_title} at {current_company} presents a mixed profile despite {years_exp} years of experience.",
        f"A borderline candidate with {years_exp} years of experience, currently at {current_company}.",
        f"Barely making the cut, this profile from {current_company} offers some relevant skills over {years_exp} years but has notable gaps."
    ]
    
    if tone == "glowing": sentences.append(rng.choice(intro_opts_glowing))
    elif tone == "solid": sentences.append(rng.choice(intro_opts_solid))
    else: sentences.append(rng.choice(intro_opts_marginal))

    all_retrieval = list(set(skills_found + prod_found))
    if all_retrieval:
        retrieval_str = ", ".join(all_retrieval[:3])
        if prod_found:
            retrieval_opts = [
                f"They perfectly match the JD's need for production vector search, explicitly mentioning {retrieval_str} in their career history.",
                f"Crucially, their profile shows hands-on production experience with retrieval technologies like {retrieval_str}.",
                f"We specifically need builders of search systems, and they have demonstrably shipped systems using {retrieval_str}."
            ]
        else:
            retrieval_opts = [
                f"Although they list {retrieval_str} in their skills, it's missing from their actual job descriptions.",
                f"They are familiar with {retrieval_str}, but the resume lacks evidence of deploying these tools to production users.",
                f"They check the box for knowing {retrieval_str}, though practical deployment context is not fully visible."
            ]
        sentences.append(rng.choice(retrieval_opts))
    else:
        sentences.append(rng.choice([
            "A major weakness is the complete absence of core retrieval keywords (like Pinecone, FAISS, or Elasticsearch).",
            "Surprisingly for this rank, there is no direct mention of vector databases or dense retrieval tools.",
            "Their profile completely lacks standard semantic search or vector DB vocabulary."
        ]))

    if eval_found:
        eval_str = ", ".join(eval_found[:2])
        eval_opts = [
            f"Their understanding of ranking evaluation is evident through mentions of {eval_str}.",
            f"They meet our requirement for rigorous measurement, noting experience with {eval_str}.",
            f"The candidate is clearly data-driven, specifically citing metrics like {eval_str}."
        ]
        sentences.append(rng.choice(eval_opts))
    else:
        if tone != "glowing":
            sentences.append(rng.choice([
                "They do not mention any explicit evaluation metrics (like NDCG or A/B testing).",
                "A notable gap is the lack of ranking evaluation keywords, making it hard to judge their rigor.",
                "There is no evidence they have formally evaluated retrieval quality."
            ]))

    if fraction > 80:
        prod_opts = [
            "Almost their entire career has been spent in product-led environments, meaning they are likely accustomed to our required ownership model.",
            "They strongly index on product company experience, making them a great cultural fit for a startup.",
            "Having built for real users rather than clients (high product-company fraction), they align well with our 0-to-1 culture."
        ]
        sentences.append(rng.choice(prod_opts))
    elif fraction < 30:
        sentences.append(rng.choice([
            "However, they have spent significant time in consulting/outsourcing, which may not translate well to our high-autonomy product culture.",
            "One hesitation is their heavy consulting background, which often implies executing against client specs rather than owning product outcomes.",
            "Their lack of pure product-company tenure is a red flag for a startup environment."
        ]))

    concerns = []
    if notice_period > 60:
        concerns.append(f"a lengthy notice period of {notice_period} days")
    if recruiter_response < 0.2:
        concerns.append(f"a historically poor recruiter response rate ({int(recruiter_response*100)}%)")
    if features.get('is_title_chaser'):
        concerns.append("a pattern of short tenures (job hopping)")
    if features.get('is_framework_only'):
        concerns.append("a reliance on wrapper frameworks without deep technical depth")
        
    if concerns:
        concerns_str = " and ".join(concerns)
        concern_opts = [
            f"We must acknowledge some risks, specifically {concerns_str}.",
            f"The primary reservations regarding this candidate are {concerns_str}.",
            f"On the downside, {concerns_str} could pose a hiring challenge."
        ]
        sentences.append(rng.choice(concern_opts))
    elif tone == "glowing":
        sentences.append(rng.choice([
            "There are virtually no red flags in their behavioral signals.",
            "They also boast excellent engagement metrics and availability.",
            "Overall, this is an incredibly clean and low-risk profile."
        ]))

    middle = sentences[1:-1] if len(sentences) > 2 else []
    if middle:
        rng.shuffle(middle)
        
    final_sentences = [sentences[0]] + middle
    if len(sentences) > 1:
        final_sentences.append(sentences[-1])
        
    final_sentences.append(f"(Score: {final_score:.4f})")
    
    return " ".join(final_sentences)


