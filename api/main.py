import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import logging

from src.dataset_loader import load_dataset
from src.embedder import Embedder
from src.vector_index import VectorIndex
from src.clustering_model import ClusteringModel
from src.cache_manager import CacheManager
from src.semantic_search import SemanticSearch
from src.query_processing import preprocess_query


logging.basicConfig(level=logging.INFO)

app=FastAPI()

logging.info("Loading dataset")

documents,labels=load_dataset("20_newsgroups")


embedder = Embedder()

EMBEDDINGS_PATH = "models/embeddings.npy"

if os.path.exists(EMBEDDINGS_PATH):
    logging.info("Loading saved embeddings")
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    logging.info("Generating embeddings")
    embeddings = embedder.encode_documents(documents)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)

dimension = len(embeddings[0])


vector_index=VectorIndex(dimension)
vector_index.add_vectors(embeddings)


cluster_model=ClusteringModel()
cluster_model.train(embeddings)


cache=CacheManager()

search_engine=SemanticSearch(documents,vector_index)


class QueryRequest(BaseModel):

    query:str


@app.post("/query")
def query(request:QueryRequest):

    logging.info(f"Incoming query: {request.query}")

    query=preprocess_query(request.query)

    query_embedding=embedder.encode_query(query)

    cluster,prob=cluster_model.dominant_cluster(query_embedding)


    hit,entry,score=cache.lookup(query_embedding,cluster)


    if hit:

        return{

            "query":query,
            "cache_status":"hit",
            "matched_query":entry["query"],
            "similarity_score":float(score),
            "result":entry["result"],
            "dominant_cluster":cluster,
            "cluster_probability":float(prob)
        }


    result=search_engine.search(query_embedding)

    cache.store(query,query_embedding,result,cluster)


    return{

        "query":query,
        "cache_status":"miss",
        "matched_query":None,
        "similarity_score":float(score),
        "result":result,
        "dominant_cluster":cluster,
        "cluster_probability":float(prob)
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message":"cache cleared"}
    