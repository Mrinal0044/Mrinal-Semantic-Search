from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def evaluate_clusters(embeddings):

    ks=range(5,21)

    bic_scores=[]

    for k in ks:

        gmm=GaussianMixture(n_components=k)

        gmm.fit(embeddings)

        bic_scores.append(gmm.bic(embeddings))


    plt.plot(ks,bic_scores)

    plt.xlabel("Clusters")

    plt.ylabel("BIC Score")

    plt.title("Cluster Selection")

    plt.show()