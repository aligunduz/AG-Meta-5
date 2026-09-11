import os
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


BASE = "materials/embedding_comparison_5shot"

FILES = {
    "Frozen Mean": (
        os.path.join(BASE, "frozen_mean_train.npz"),
        os.path.join(BASE, "frozen_mean_val.npz"),
    ),
    "Task2Vec": (
        os.path.join(BASE, "task2vec_train.npz"),
        os.path.join(BASE, "task2vec_val.npz"),
    ),
}


def purity_score(y_true, y_cluster):
    total = 0
    for c in np.unique(y_cluster):
        labels = y_true[y_cluster == c]
        _, counts = np.unique(labels, return_counts=True)
        total += counts.max()
    return total / len(y_true)


def nearest_domain_centroid_accuracy(x_train, y_train, x_val, y_val):
    domains = sorted(np.unique(y_train))

    centroids = []
    for d in domains:
        c = x_train[y_train == d].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)

    centroids = np.stack(centroids)

    # embeddings zaten L2-normalized:
    # cosine similarity = dot product
    similarity = x_val @ centroids.T

    pred_idx = similarity.argmax(axis=1)
    preds = np.array([domains[i] for i in pred_idx])

    overall = np.mean(preds == y_val)

    by_domain = {}
    for d in domains:
        mask = y_val == d
        by_domain[d] = np.mean(preds[mask] == y_val[mask])

    return overall, by_domain


def intra_inter_ratio(x, y):
    sim = x @ x.T
    dist = 1.0 - sim

    intra = []
    inter = []

    n = len(y)

    for i in range(n):
        for j in range(i + 1, n):
            if y[i] == y[j]:
                intra.append(dist[i, j])
            else:
                inter.append(dist[i, j])

    intra_mean = np.mean(intra)
    inter_mean = np.mean(inter)

    return intra_mean, inter_mean, intra_mean / inter_mean


for method, (train_path, val_path) in FILES.items():

    train = np.load(train_path)
    val = np.load(val_path)

    x_train = train["embeddings"].astype(np.float64)
    y_train = train["domains"]

    x_val = val["embeddings"].astype(np.float64)
    y_val = val["domains"]

    # Aynı preprocessing ve aynı metric:
    # L2-normalization + cosine geometry.
    x_train = normalize(x_train, norm="l2")
    x_val = normalize(x_val, norm="l2")

    # -----------------------------------------
    # 1. Unsupervised K=9 clustering
    # -----------------------------------------
    km = KMeans(
        n_clusters=9,
        random_state=42,
        n_init=50,
    )

    clusters = km.fit_predict(x_train)

    purity = purity_score(y_train, clusters)
    nmi = normalized_mutual_info_score(
        y_train,
        clusters,
    )
    ari = adjusted_rand_score(
        y_train,
        clusters,
    )

    silhouette = silhouette_score(
        x_train,
        clusters,
        metric="cosine",
    )

    # -----------------------------------------
    # 2. Train-domain centers -> val routing
    # -----------------------------------------
    routing_acc, routing_by_domain = (
        nearest_domain_centroid_accuracy(
            x_train,
            y_train,
            x_val,
            y_val,
        )
    )

    # -----------------------------------------
    # 3. Intra / inter domain separation
    # -----------------------------------------
    intra, inter, ratio = intra_inter_ratio(
        x_train,
        y_train,
    )

    print()
    print("=" * 70)
    print(method)
    print("=" * 70)

    print(f"Embedding dim       : {x_train.shape[1]}")
    print(f"Purity              : {purity:.4f}")
    print(f"NMI                 : {nmi:.4f}")
    print(f"ARI                 : {ari:.4f}")
    print(f"Silhouette          : {silhouette:.4f}")

    print()
    print(f"VAL routing accuracy: {routing_acc:.4f}")

    print("\nVAL routing by domain:")
    for d, acc in routing_by_domain.items():
        print(f"  {d:10s}: {acc:.4f}")

    print()
    print(f"Intra-domain dist   : {intra:.6f}")
    print(f"Inter-domain dist   : {inter:.6f}")
    print(f"Intra / Inter ratio : {ratio:.4f}")