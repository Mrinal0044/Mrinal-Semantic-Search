Semantic Search System with Cluster-Aware Semantic Cache

OVERVIEW

This project implements a lightweight semantic search system built using the 20 Newsgroups dataset (~20,000 documents).
Instead of traditional keyword search, this system retrieves documents using semantic similarity based on vector embeddings.
Additionally, it introduces a cluster-aware semantic cache that detects similar queries even if they are phrased differently, reducing redundant computations.
The system integrates multiple machine learning and system design components:

1. Sentence Transformer embeddings
2. FAISS vector similarity search
3. Gaussian Mixture Model (GMM) clustering
4. Cluster-aware semantic caching
5. FastAPI REST service

The goal of this project is to demonstrate how semantic retrieval, clustering, and caching can work together to build an efficient search system.

SYSTEM ARCHITECTURE

User Query
     ↓
Query Preprocessing
     ↓
Query Embedding (Sentence Transformers)
     ↓
Cluster Detection (Gaussian Mixture Model)
     ↓
Semantic Cache Lookup
     ↓
Cache Hit → Return Cached Result
     ↓
Cache Miss → Vector Search using FAISS
     ↓
Retrieve Most Relevant Document
     ↓
Store Query and Result in Cache
     ↓
Return Result to User

EMBEDDING MODEL

The system uses the Sentence Transformers model:
all-MiniLM-L6-v2

Reasons for choosing this model:
- Lightweight
- Fast embedding generation
-Strong semantic similarity performance
-Widely used in real-world NLP systems
-Each document is converted into a 384-dimensional embedding vector.

