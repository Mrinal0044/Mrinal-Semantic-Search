# 🔍 Semantic Search System with Cluster-Aware Semantic Cache

A lightweight yet production-ready semantic search engine built on the **20 Newsgroups dataset** (~20,000 documents). Instead of keyword matching, this system retrieves documents by meaning — and uses a cluster-aware semantic cache to make repeated queries lightning fast.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
  - [Dataset](#dataset)
  - [Embedding Model](#embedding-model)
  - [Vector Database (FAISS)](#vector-database-faiss)
  - [Fuzzy Clustering (GMM)](#fuzzy-clustering-gmm)
  - [Semantic Cache](#semantic-cache)
- [API Reference](#api-reference)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [Analysis Scripts](#analysis-scripts)
- [Example Queries](#example-queries)
- [Performance Optimizations](#performance-optimizations)
- [Tech Stack](#tech-stack)

---

## Overview

This project combines several machine learning and systems design components into a cohesive search service:

| Component | Technology |
|---|---|
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS |
| Clustering | Gaussian Mixture Models (GMM) |
| Cache Layer | Cluster-aware semantic cache |
| API Server | FastAPI |

The goal is to demonstrate how **semantic retrieval**, **probabilistic clustering**, and **intelligent caching** can be combined to build an efficient, real-world search system.

---

## Architecture

```
User Query
    │
    ▼
Query Preprocessing
    │
    ▼
Query Embedding               ← Sentence Transformers (384-dim vector)
    │
    ▼
Cluster Detection             ← Gaussian Mixture Model (GMM)
    │
    ▼
Semantic Cache Lookup         ← Cosine similarity against cached embeddings
    │
    ├── Cache HIT  ──────────────────────────────────► Return cached result
    │
    └── Cache MISS
            │
            ▼
       FAISS Vector Search    ← Nearest neighbor search across ~20k documents
            │
            ▼
       Return top document
            │
            ▼
       Store in semantic cache (keyed by cluster)
```

---

## Project Structure

```
semantic-search-system/
│
├── api/
│   └── main.py                  # FastAPI app — endpoints and request handling
│
├── src/
│   ├── dataset_loader.py        # Loads the 20 Newsgroups dataset
│   ├── embedder.py              # Generates sentence embeddings
│   ├── vector_index.py          # Builds and queries the FAISS index
│   ├── clustering_model.py      # Trains and applies the GMM clustering model
│   ├── cache_manager.py         # Manages the cluster-aware semantic cache
│   ├── semantic_search.py       # Orchestrates the full search pipeline
│   └── query_processing.py      # Preprocesses and normalizes queries
│
├── analysis/
│   ├── cluster_selection.py     # BIC analysis to find optimal cluster count
│   └── cluster_visualization.py # UMAP 2D projection of document embeddings
│
├── models/
│   └── embeddings.npy           # Cached document embeddings (generated on first run)
│
├── 20_newsgroups/               # Raw dataset directory
├── requirements.txt
└── README.md
```

---

## Key Components

### Dataset

The system uses the **20 Newsgroups** dataset — approximately 20,000 news documents across 20 topic categories.

Example categories:

- `alt.atheism`
- `comp.graphics`
- `rec.autos`
- `sci.space`
- `talk.politics.misc`

Although documents have labeled categories, semantic content overlaps significantly across topics, making this an ideal dataset for fuzzy, probabilistic clustering.

---

### Embedding Model

Documents and queries are encoded using:

```
all-MiniLM-L6-v2  (Sentence Transformers)
```

This model produces **384-dimensional vector embeddings** that capture semantic meaning rather than surface-level keywords.

Why this model:
- Lightweight and fast — suitable for local inference
- Strong semantic similarity performance
- Widely used in production NLP systems

---

### Vector Database (FAISS)

Similarity search is powered by **FAISS** (Facebook AI Similarity Search), which performs fast approximate nearest neighbor lookups across the full document corpus.

Search pipeline:

```
Query → Embedding (384-dim) → FAISS Index → Top-K Matching Documents
```

---

### Fuzzy Clustering (GMM)

The semantic structure of the corpus is modeled using **Gaussian Mixture Models (GMM)** rather than hard clustering (e.g., K-Means).

GMM assigns **probabilistic cluster memberships**, meaning a document can belong to multiple topics simultaneously:

```
Document: "Gun legislation debate"

Cluster probabilities:
  Politics  → 0.55
  Religion  → 0.18
  Society   → 0.27
```

This better reflects the real-world overlap between news topics.

**Cluster count selection** is done via Bayesian Information Criterion (BIC) analysis — the number of clusters minimizing the BIC score is selected as optimal. See `analysis/cluster_selection.py`.

---

### Semantic Cache

Traditional caches fail when the same question is phrased differently:

```
"What is atheism?"          ← exact cache key: miss
"Explain atheistic beliefs" ← same meaning, different words: miss (traditional cache)
```

This system implements a **semantic cache** that compares query embeddings using **cosine similarity**. If an incoming query is sufficiently similar to a previously-seen query, the cached result is returned — no vector search needed.

**Cache lookup flow:**

```
Incoming query
    │
    ▼
Generate embedding
    │
    ▼
Compare cosine similarity against cached query embeddings
    │
    ├── similarity > threshold  ──► Cache HIT — return cached result
    └── similarity ≤ threshold  ──► Cache MISS — run FAISS search
```

**Cluster-aware optimization:**

Cached queries are grouped by their dominant cluster:

```
cluster_0 → [cached query embeddings]
cluster_1 → [cached query embeddings]
...
```

This reduces lookup complexity from **O(n)** across all cached entries to **O(cluster_size)** — a meaningful speedup as the cache grows.

---

## API Reference

The system exposes a FastAPI REST service. Interactive docs available at `http://127.0.0.1:8000/docs`.

---

### `POST /query`

Run a semantic search query.

**Request body:**
```json
{
  "query": "What is atheism?"
}
```

**Response:**
```json
{
  "query": "what is atheism",
  "cache_status": "miss",
  "matched_query": null,
  "similarity_score": 0.0,
  "result": "...retrieved document text...",
  "dominant_cluster": 3,
  "cluster_probability": 0.81
}
```

| Field | Description |
|---|---|
| `cache_status` | `"hit"` or `"miss"` |
| `matched_query` | The cached query that triggered the hit (if applicable) |
| `similarity_score` | Cosine similarity to the matched cached query |
| `result` | The retrieved document text |
| `dominant_cluster` | GMM cluster the query was assigned to |
| `cluster_probability` | Probability of dominant cluster membership |

---

### `GET /cache/stats`

Return cache performance statistics.

**Response:**
```json
{
  "total_entries": 2,
  "hit_count": 1,
  "miss_count": 1,
  "hit_rate": 0.5
}
```

---

### `DELETE /cache`

Clear all cached queries and reset statistics.

---

## Setup & Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/semantic-search-system.git
cd semantic-search-system
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv venv
```

### Step 3 — Activate the virtual environment

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 5 — Start the FastAPI server

```bash
python -m uvicorn api.main:app --reload
```

The server will start at:

```
http://127.0.0.1:8000
```

### Step 6 — Open the interactive API docs

```
http://127.0.0.1:8000/docs
```

This provides a full Swagger UI for exploring and testing all endpoints.

> **Note on first run:** On the first run, the system generates embeddings for all ~20,000 documents. This takes approximately **1–2 minutes**. Embeddings are saved to `models/embeddings.npy` and reloaded instantly on all subsequent runs (~3 seconds).

---

## Analysis Scripts

### Cluster count selection (BIC analysis)

Determines the optimal number of GMM clusters by minimizing the BIC score:

```bash
python analysis/cluster_selection.py
```

### Cluster visualization (UMAP projection)

Projects document embeddings into 2D space using UMAP for visual exploration of the semantic cluster structure:

```bash
python analysis/cluster_visualization.py
```


This produces a scatter plot showing semantic topic clusters and their boundaries across the corpus.

![Semantic Clusters of 20 Newsgroups Dataset](https://github.com/user-attachments/assets/1ed9a828-1137-4c15-87e0-7828354d4426)

> Each color represents a GMM cluster. Notice the tight isolated clusters (e.g., pink at top, orange at bottom-left) representing highly cohesive topics, while the dense overlapping region in the center reflects semantically similar or cross-topical newsgroup content.


---

## Example Queries

Use these pairs to observe semantic cache behavior — submit the first query, then the second, and watch the `cache_status` change from `miss` to `hit`:

| Query 1 | Query 2 (semantically equivalent) |
|---|---|
| What is atheism? | Explain atheistic beliefs |
| What are gun laws? | Explain firearm legislation |
| What is space exploration? | Explain missions to outer space |
| How do computers work? | Explain computer hardware |

---

## Performance Optimizations

### Embedding caching

Generating embeddings for the full corpus is expensive. On the first run, embeddings are computed and saved to disk:

```
First run:       ~1–2 minutes   (compute + save)
Subsequent runs: ~3 seconds     (load from disk)
```

This mirrors standard ML deployment practices where pre-computed representations are persisted and reused.

### Cluster-aware cache lookup

Partitioning cached queries by cluster turns a linear scan of all cached entries into a bounded lookup within a single cluster — making the cache increasingly efficient as it grows.

### Query logging

All incoming requests are logged for monitoring and debugging:

```
INFO: Incoming query: what is atheism
INFO: Cache miss
INFO: Performing vector search
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `sentence-transformers` | Document and query embeddings |
| `faiss-cpu` | Approximate nearest neighbor vector search |
| `scikit-learn` | GMM clustering, dataset loading |
| `numpy` | Embedding storage and operations |
| `umap-learn` | Dimensionality reduction for visualization |
| `matplotlib` | Cluster visualization plots |

---


