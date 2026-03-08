from sklearn.mixture import GaussianMixture
import numpy as np

class ClusteringModel:

    def __init__(self,n_clusters=10):

        self.model=GaussianMixture(n_components=n_clusters)

    def train(self,embeddings):

        self.model.fit(embeddings)

    def dominant_cluster(self,embedding):

        probs=self.model.predict_proba([embedding])[0]

        return int(np.argmax(probs)),max(probs)