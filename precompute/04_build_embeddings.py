import pickle
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import os
    os.system('pip install sentence-transformers')
    from sentence_transformers import SentenceTransformer

def main():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    jd_query = """Production embedding retrieval system vector database hybrid search semantic search
NDCG MRR MAP ranking evaluation A/B testing offline evaluation
Applied ML product company startup shipped real users scale
NLP information retrieval sentence transformers Pinecone Qdrant Elasticsearch FAISS
Python recommendation system ranking system search
Pune Noida India available notice period"""
    
    print("Encoding JD...")
    jd_embedding = model.encode(jd_query)
    
    with open('precompute/artifacts/candidates_parsed.pkl', 'rb') as f:
        candidates = pickle.load(f)
        
    print(f"Encoding {len(candidates)} candidates...")
    
    texts = []
    for cand in candidates:
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
        
        full_text = " ".join([summary, headline, work_text, work_titles, skills_text])
        texts.append(full_text)
        
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
    
    np.save('precompute/artifacts/embeddings.npy', embeddings)
    np.save('precompute/artifacts/jd_embedding.npy', jd_embedding)
    
    print("Done building embeddings.")

if __name__ == '__main__':
    main()
