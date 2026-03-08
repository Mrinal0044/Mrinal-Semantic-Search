Semantic Search System with Cluster-Aware Semantic Cache

Overview
This project implements a lightweight semantic search system built using the 20 Newsgroups dataset (~20,000 documents).

The system retrieves documents based on semantic similarity rather than keyword matching and improves efficiency through a cluster-aware semantic caching layer.

The system integrates several machine learning and system design components:

Sentence Transformer embeddings

FAISS vector database

Gaussian Mixture Model (GMM) clustering

Semantic cache

FastAPI service

The goal is to demonstrate how semantic retrieval, clustering, and caching can be combined to build an efficient search system.

System Architecture
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
If cache hit → return cached result
    ↓
Else → Vector Search using FAISS
    ↓
Return most relevant document
    ↓
Store result in semantic cache


Dataset

The system uses the 20 Newsgroups dataset, which contains approximately 20,000 documents across 20 categories.

Example categories include:

alt.atheism
comp.graphics
rec.autos
sci.space
talk.politics.misc

Although each document has a labeled category, the semantic content often overlaps across topics, making this dataset suitable for fuzzy clustering approaches.

Embedding Model

The system uses the Sentence Transformers model:

all-MiniLM-L6-v2

Reasons for choosing this model:

lightweight

fast embedding generation

strong semantic similarity performance

commonly used in production NLP systems

Each document is converted into a 384-dimensional vector embedding.

Vector Database

To perform efficient similarity search, the system uses FAISS (Facebook AI Similarity Search).

FAISS allows fast nearest neighbor search across thousands of vector embeddings.

Search pipeline:

Query → Embedding → FAISS Search → Top Matching Documents
Fuzzy Clustering

The semantic structure of the dataset is modeled using Gaussian Mixture Models (GMM).

Unlike traditional clustering, GMM provides probabilistic cluster membership, meaning documents can belong to multiple clusters.

Example:

Document: "Gun legislation debate"

Cluster membership:
Politics → 0.55
Religion → 0.18
Society → 0.27

This better represents semantic topic overlap.

Cluster Selection

The number of clusters was determined using Bayesian Information Criterion (BIC) analysis.

The cluster count minimizing the BIC score was selected as the optimal configuration.

Script used:

analysis/cluster_selection.py
Cluster Visualization

To visualize the semantic structure of the dataset, embeddings are projected into 2D space using UMAP.

This visualization helps identify:

semantic topic clusters

overlapping topic boundaries

dense document groups

Script used:

analysis/cluster_visualization.py

Example visualization:

Document Embeddings
      ↓
UMAP Dimensionality Reduction
      ↓
2D Semantic Cluster Visualization
Semantic Cache

Traditional caching fails when the same question is asked in different wording.

Example:

"What is atheism?"
"Explain atheistic beliefs"

The system implements a semantic cache that compares query embeddings using cosine similarity.

Process:

New Query
   ↓
Generate embedding
   ↓
Compare with cached query embeddings
   ↓
If similarity > threshold → cache hit
Cache Optimization

The cache is cluster-aware, meaning queries are grouped by their dominant cluster.

cluster_id → cached queries

This reduces lookup complexity from:

O(n) → O(cluster_size)

which improves performance as cache size grows.

API Endpoints

The system exposes a FastAPI REST service.

Query Endpoint
POST /query

Example request:

{
  "query": "What is atheism?"
}

Example response:

{
  "query": "what is atheism",
  "cache_status": "miss",
  "matched_query": null,
  "similarity_score": 0.0,
  "result": "...document text...",
  "dominant_cluster": 3,
  "cluster_probability": 0.81
}
Cache Statistics
GET /cache/stats

Example response:

{
  "total_entries": 2,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
Clear Cache
DELETE /cache

Clears all cached queries and statistics.

Improvement 1 — Embedding Caching

Generating embeddings for ~20,000 documents can take several minutes.

To improve performance, embeddings are saved locally and reused on subsequent runs.

Example logic:

If embeddings file exists
    load embeddings
Else
    generate embeddings
    save embeddings

Benefits:

First run → ~1–2 minutes
Subsequent runs → ~3 seconds

This mirrors real-world ML system deployment practices.

Improvement 2 — Query Logging

The system includes query logging to monitor incoming requests.

Example log:

INFO: Incoming query: what is atheism
INFO: Cache miss
INFO: Performing vector search

This improves:

system monitoring

debugging

production readiness

Running the Project
1 Create Virtual Environment
python3 -m venv venv
2 Activate Environment
source venv/bin/activate
3 Install Dependencies
pip install -r requirements.txt
4 Run FastAPI Server
python -m uvicorn api.main:app --reload
API Documentation

Open:

http://127.0.0.1:8000/docs

This provides an interactive Swagger interface for testing endpoints.

Example Queries for Testing
What is atheism?
Explain atheistic beliefs
What are gun laws?
Explain firearm legislation
What is space exploration?
Explain missions to outer space

These demonstrate the semantic cache behavior.

Technologies Used

Python

FastAPI

Sentence Transformers

FAISS

Scikit-learn

NumPy

UMAP

Matplotlib

Project Structure
semantic-search-system
│
├── 20_newsgroups
├── analysis
│   ├── cluster_selection.py
│   └── cluster_visualization.py
│
├── api
│   └── main.py
│
├── src
│   ├── dataset_loader.py
│   ├── embedder.py
│   ├── vector_index.py
│   ├── clustering_model.py
│   ├── cache_manager.py
│   ├── semantic_search.py
│   └── query_processing.py
│
├── models
│   └── embeddings.npy
│
├── requirements.txt
└── README.md
Key Features

✔ Semantic document retrieval
✔ Vector similarity search using FAISS
✔ Fuzzy clustering of documents
✔ Cluster-aware semantic cache
✔ FastAPI REST API
✔ Query logging
✔ Embedding caching
✔ Cluster visualization