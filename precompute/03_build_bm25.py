import json
import pickle
import numpy as np

def compute_bm25(corpus, query):
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        doc_scores = bm25.get_scores(tokenized_query)
        # Ensure no negative scores (happens in Okapi if term in >50% of docs)
        doc_scores = np.maximum(doc_scores, 0.0)
        # Normalize to 0-1
        if len(doc_scores) > 0 and max(doc_scores) > 0:
            doc_scores = doc_scores / max(doc_scores)
        return doc_scores

    except ImportError:
        # Fallback to simple TF-IDF if rank_bm25 is missing
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        Q = vectorizer.transform([query])
        scores = cosine_similarity(X, Q).flatten()
        if len(scores) > 0 and max(scores) > 0:
            scores = scores / max(scores)
        return scores

def main():
    jd_query = """Production embedding retrieval system vector database hybrid search semantic search
NDCG MRR MAP ranking evaluation A/B testing offline evaluation
Applied ML product company startup shipped real users scale
NLP information retrieval sentence transformers Pinecone Qdrant Elasticsearch FAISS
Python recommendation system ranking system search
Pune Noida India available notice period"""
    
    with open('precompute/artifacts/candidates_parsed.pkl', 'rb') as f:
        candidates = pickle.load(f)
        
    corpus = []
    candidate_ids = []
    for cand in candidates:
        candidate_ids.append(cand.get('candidate_id'))
        profile = cand.get('profile', {})
        career = cand.get('career_history', [])
        skills = cand.get('skills', [])
        
        work_text = " ".join([c.get('description', '') for c in career if c.get('description')])
        work_titles = " ".join([c.get('title', '') for c in career if c.get('title')])
        skills_text = " ".join([s.get('name', '') for s in skills if s.get('name')])
        summary = profile.get('summary', '')
        if not summary: summary = ''
        headline = profile.get('headline', '')
        if not headline: headline = ''
        
        full_text = " ".join([summary, headline, work_text, work_titles, skills_text]).lower()
        # Basic tokenization
        import re
        full_text = re.sub(r'[^a-z0-9\s]', ' ', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        corpus.append(full_text)
        
    jd_query_clean = jd_query.lower()
    import re
    jd_query_clean = re.sub(r'[^a-z0-9\s]', ' ', jd_query_clean)
    jd_query_clean = re.sub(r'\s+', ' ', jd_query_clean).strip()
    
    scores = compute_bm25(corpus, jd_query_clean)
    
    bm25_dict = {cid: float(score) for cid, score in zip(candidate_ids, scores)}
    
    with open('precompute/artifacts/bm25_scores.pkl', 'wb') as f:
        pickle.dump(bm25_dict, f)
        
    print(f"Computed BM25 scores for {len(bm25_dict)} candidates.")

if __name__ == '__main__':
    main()
