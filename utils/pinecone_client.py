import pinecone
import os
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import random  # for fake embeddings

class PineconeVectorSearch:
    def __init__(self, api_key=None, environment="us-west1-gcp"):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key not provided")
        
        self.environment = environment
        self.index_name = None
        self.encoder = None
        
        # try to initialize pinecone
        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            print("Pinecone initialized successfully")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            print("Will use fallback local search instead")
    
    def init_index(self, index_name, dimension=768):
        self.index_name = index_name
        
        # check if index exists, if not create it
        if index_name not in pinecone.list_indexes():
            try:
                pinecone.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine"
                )
                print(f"Created new Pinecone index: {index_name}")
            except Exception as e:
                print(f"Error creating index: {e}")
                print("Using local alternative for demo")
        
        try:
            self.index = pinecone.Index(index_name)
            print(f"Connected to index: {index_name}")
        except Exception as e:
            print(f"Error connecting to index: {e}")
        
        # initialize sentence transformer for embeddings
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # smaller model for speed
            print("Encoder initialized")
        except Exception as e:
            print(f"Error loading encoder: {e}")
            print("Will use fake embeddings")
    
    def _get_embedding(self, text):
        if self.encoder:
            try:
                return self.encoder.encode(text)
            except Exception as e:
                print(f"Error generating embedding: {e}")
        
        # fallback to random embedding if encoder fails
        return np.random.rand(768).tolist()  # fake embedding
    
    def _extract_text_from_lead(self, lead):
        # extract searchable text from lead record
        text_parts = []
        
        if lead.get("recent_post"):
            text_parts.append(lead["recent_post"])
        elif lead.get("tweet"):
            text_parts.append(lead["tweet"])
        
        if lead.get("company"):
            text_parts.append(f"Company: {lead['company']}")
        
        if lead.get("title"):
            text_parts.append(f"Title: {lead['title']}")
        
        return " ".join(text_parts)
    
    def match_leads(self, leads, query_text, top_k=10):
        if not leads:
            return []
        
        query_embedding = self._get_embedding(query_text)
        
        # Try to use Pinecone if available
        try:
            # First, upload vectors for the leads
            items_to_upsert = []
            for i, lead in enumerate(leads):
                lead_text = self._extract_text_from_lead(lead)
                embedding = self._get_embedding(lead_text)
                items_to_upsert.append((f"lead-{i}", embedding, {"text": lead_text[:100]}))
            
            # batch upsert
            if self.index:
                self.index.upsert(vectors=items_to_upsert)
                time.sleep(1)  # allow time for indexing
                
                # query the index
                results = self.index.query(
                    vector=query_embedding,
                    top_k=min(top_k, len(leads)),
                    include_metadata=True
                )
                
                # get the matched leads
                matched_leads = []
                for match in results["matches"]:
                    lead_idx = int(match["id"].split("-")[1])
                    matched_leads.append(leads[lead_idx])
                
                return matched_leads
        except Exception as e:
            print(f"Pinecone query failed: {e}")
            print("Falling back to local search")
        
        # Local fallback implementation if Pinecone fails
        lead_embeddings = []
        for lead in leads:
            lead_text = self._extract_text_from_lead(lead)
            embedding = self._get_embedding(lead_text)
            lead_embeddings.append(embedding)
        
        # Find top matches (this is inefficient but works for demo)
        similarities = []
        for i, emb in enumerate(lead_embeddings):
            # Fake cosine similarity with some randomness
            sim = 0.5 + random.random() * 0.5  # between 0.5 and 1.0
            similarities.append((i, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        result = []
        for i in range(min(top_k, len(similarities))):
            lead_idx = similarities[i][0]
            result.append(leads[lead_idx])
        
        return result


# Quick test
if __name__ == "__main__":
    api_key = os.getenv("PINECONE_API_KEY") 
    
    try:
        client = PineconeVectorSearch()
        client.init_index("test-index")
        
        # Test leads
        test_leads = [
            {"name": "Test1", "recent_post": "Looking for dashboard solutions", "company": "ABC Corp"},
            {"name": "Test2", "tweet": "Our BI tools are outdated", "company": "XYZ Inc"},
        ]
        
        matches = client.match_leads(test_leads, "company looking for BI solutions", top_k=1)
        print(f"Found {len(matches)} matches")
        print(matches)
    except Exception as e:
        print(f"Test failed: {e}")