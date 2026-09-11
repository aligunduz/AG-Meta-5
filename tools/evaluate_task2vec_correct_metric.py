import os
import numpy as np
import torch

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

BASE = "materials/embedding_comparison_5shot"

TRAIN = os.path.join(BASE, "task2vec_train.npz")
VAL = os.path.join(BASE, "task2vec_val.npz")


def purity_score(y_true, y_cluster):
    total = 0

    for c in np.unique(y_cluster):
        labels = y_true[y_cluster == c]
        _, counts = np.unique(labels, return_counts=True)
        total += counts.max()

    return total / len(y_true)


def task2vec_distance_matrix(a, b, device="cuda", chunk_size=8):
    """
    Official symmetric Task2Vec distance:

    d_sym(Fa,Fb) =
      cosine_distance(
        Fa/(Fa+Fb),
        Fb/(Fa+Fb)
      )
    """

    A = torch.tensor(
        a,
        dtype=torch.float32,
        device=device,
    )

    B = torch.tensor(
        b,
        dtype=torch.float32,
        device=device,
    )

    result = np.zeros(
        (len(a), len(b)),
        dtype=np.float32,
    )

    eps = 1e-12

    for start in range(0, len(a), chunk_size):
        end = min(start + chunk_size, len(a))

        aa = A[start:end, None, :]
        bb = B[None, :, :]

        denom = aa + bb + eps

        na = aa / denom
        nb = bb / denom

        dot = (na * nb).sum(dim=-1)

        norm_a = torch.sqrt(
            (na * na).sum(dim=-1) + eps
        )

        norm_b = torch.sqrt(
            (nb * nb).sum(dim=-1) + eps
        )

        cosine = dot / (norm_a * norm_b)

        dist = 1.0 - cosine

        result[start:end] = (
            dist.detach().cpu().numpy()
        )

    return result


train = np.load(TRAIN)
val = np.load(VAL)

x_train = train["embeddings"]
y_train = train["domains"]

x_val = val["embeddings"]
y_val = val["domains"]

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# --------------------------------------------------
# 1. Train pairwise Task2Vec distance
# --------------------------------------------------

print("Train distance matrix hesaplanıyor...")

D_train = task2vec_distance_matrix(
    x_train,
    x_train,
    device=device,
)

np.fill_diagonal(D_train, 0.0)

# --------------------------------------------------
# 2. K=9 clustering
# --------------------------------------------------

clusterer = AgglomerativeClustering(
    n_clusters=9,
    metric="precomputed",
    linkage="average",
)

clusters = clusterer.fit_predict(D_train)

purity = purity_score(
    y_train,
    clusters,
)

nmi = normalized_mutual_info_score(
    y_train,
    clusters,
)

ari = adjusted_rand_score(
    y_train,
    clusters,
)

silhouette = silhouette_score(
    D_train,
    clusters,
    metric="precomputed",
)

# --------------------------------------------------
# 3. Her domain için Task2Vec medoid
# --------------------------------------------------

domains = sorted(np.unique(y_train))

medoids = []

print("Domain medoidları hesaplanıyor...")

for domain in domains:

    idx = np.where(y_train == domain)[0]

    sub = D_train[
        np.ix_(idx, idx)
    ]

    mean_dist = sub.mean(axis=1)

    medoid_local = mean_dist.argmin()

    medoid_global = idx[medoid_local]

    medoids.append(
        x_train[medoid_global]
    )

medoids = np.stack(medoids)

# --------------------------------------------------
# 4. Validation routing
# --------------------------------------------------

print("Validation routing hesaplanıyor...")

D_val = task2vec_distance_matrix(
    x_val,
    medoids,
    device=device,
)

pred_idx = D_val.argmin(axis=1)

pred = np.array([
    domains[i]
    for i in pred_idx
])

routing_acc = np.mean(
    pred == y_val
)

routing_by_domain = {}

for domain in domains:

    mask = y_val == domain

    routing_by_domain[domain] = np.mean(
        pred[mask] == y_val[mask]
    )

# --------------------------------------------------
# 5. Intra / inter
# --------------------------------------------------

intra = []
inter = []

for i in range(len(y_train)):
    for j in range(i + 1, len(y_train)):

        if y_train[i] == y_train[j]:
            intra.append(D_train[i, j])
        else:
            inter.append(D_train[i, j])

intra = np.asarray(intra)
inter = np.asarray(inter)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print()
print("=" * 70)
print("Task2Vec — Official Symmetric Distance")
print("=" * 70)

print(f"Purity              : {purity:.4f}")
print(f"NMI                 : {nmi:.4f}")
print(f"ARI                 : {ari:.4f}")
print(f"Silhouette          : {silhouette:.4f}")

print()
print(
    f"VAL routing accuracy: "
    f"{routing_acc:.4f}"
)

print("\nVAL routing by domain:")

for domain in domains:
    print(
        f"  {domain:10s}: "
        f"{routing_by_domain[domain]:.4f}"
    )

print()

print(
    f"Intra-domain dist   : "
    f"{intra.mean():.6f}"
)

print(
    f"Inter-domain dist   : "
    f"{inter.mean():.6f}"
)

print(
    f"Intra / Inter ratio : "
    f"{intra.mean() / inter.mean():.4f}"
)