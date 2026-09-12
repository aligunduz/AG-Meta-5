import os
import numpy as np
import torch

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix,
)

BASE = "materials/embedding_comparison_5shot"

DOMAINS = [
    "ACT_40",
    "ACT_410",
    "SPT",
    "RESISC",
    "RSD",
    "RSICB",
    "TEX",
    "TEX_ALOT",
    "TEX_DTD",
]

FAMILIES = {
    "ACT_40": "A",
    "ACT_410": "A",
    "SPT": "A",
    "RESISC": "B",
    "RSD": "B",
    "RSICB": "B",
    "TEX": "C",
    "TEX_ALOT": "C",
    "TEX_DTD": "C",
}


def purity_score(y_true, y_cluster):
    total = 0

    for c in np.unique(y_cluster):
        labels = y_true[y_cluster == c]
        _, counts = np.unique(labels, return_counts=True)
        total += counts.max()

    return total / len(y_true)


def family_accuracy(y_true, y_pred):
    true_family = np.array([FAMILIES[d] for d in y_true])
    pred_family = np.array([FAMILIES[d] for d in y_pred])

    return np.mean(true_family == pred_family)


def print_confusion(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=DOMAINS,
    )

    print("\n9x9 Confusion Matrix")
    print("rows=true, cols=pred\n")

    header = "true\\pred".ljust(11)

    for d in DOMAINS:
        header += d[:8].rjust(9)

    print(header)

    for i, d in enumerate(DOMAINS):
        row = d[:10].ljust(11)

        for value in cm[i]:
            row += str(value).rjust(9)

        print(row)


def print_domain_accuracy(y_true, y_pred):
    print("\nDomain accuracy:")

    for d in DOMAINS:
        mask = y_true == d

        acc = np.mean(
            y_pred[mask] == y_true[mask]
        )

        print(f"  {d:10s}: {acc:.4f}")


def print_family_accuracy(y_true, y_pred):
    print("\nFamily accuracy:")

    for family in ["A", "B", "C"]:
        mask = np.array(
            [FAMILIES[d] == family for d in y_true]
        )

        true_family = np.array(
            [FAMILIES[d] for d in y_true[mask]]
        )

        pred_family = np.array(
            [FAMILIES[d] for d in y_pred[mask]]
        )

        acc = np.mean(
            true_family == pred_family
        )

        print(f"  {family}: {acc:.4f}")


# --------------------------------------------------
# Frozen Mean
# --------------------------------------------------

def evaluate_frozen_mean():
    print("\n" + "=" * 80)
    print("FROZEN MEAN")
    print("=" * 80)

    train = np.load(
        os.path.join(
            BASE,
            "frozen_mean_train.npz",
        )
    )

    val = np.load(
        os.path.join(
            BASE,
            "frozen_mean_val.npz",
        )
    )

    x_train = train["embeddings"].astype(np.float64)
    y_train = train["domains"]

    x_val = val["embeddings"].astype(np.float64)
    y_val = val["domains"]

    # L2 normalize
    x_train /= (
        np.linalg.norm(
            x_train,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )

    x_val /= (
        np.linalg.norm(
            x_val,
            axis=1,
            keepdims=True,
        )
        + 1e-12
    )

    # --------------------------------------
    # K=9 clustering
    # --------------------------------------

    km = KMeans(
        n_clusters=9,
        random_state=42,
        n_init=50,
    )

    clusters = km.fit_predict(x_train)

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

    # --------------------------------------
    # 9 domain centroid
    # --------------------------------------

    centroids = []

    for d in DOMAINS:
        c = x_train[
            y_train == d
        ].mean(axis=0)

        c /= (
            np.linalg.norm(c)
            + 1e-12
        )

        centroids.append(c)

    centroids = np.stack(centroids)

    similarity = (
        x_val @ centroids.T
    )

    pred_idx = similarity.argmax(axis=1)

    y_pred = np.array(
        [DOMAINS[i] for i in pred_idx]
    )

    acc9 = np.mean(
        y_pred == y_val
    )

    acc3 = family_accuracy(
        y_val,
        y_pred,
    )

    print(f"Purity : {purity:.4f}")
    print(f"NMI    : {nmi:.4f}")
    print(f"ARI    : {ari:.4f}")

    print()
    print(f"Acc9   : {acc9:.4f}")
    print(f"Acc3   : {acc3:.4f}")

    print_domain_accuracy(
        y_val,
        y_pred,
    )

    print_family_accuracy(
        y_val,
        y_pred,
    )

    print_confusion(
        y_val,
        y_pred,
    )


# --------------------------------------------------
# Task2Vec distance
# --------------------------------------------------

def task2vec_distance_matrix(
    a,
    b,
    device,
    chunk_size=8,
):
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

    for start in range(
        0,
        len(a),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(a),
        )

        aa = A[
            start:end,
            None,
            :
        ]

        bb = B[
            None,
            :,
            :
        ]

        denom = aa + bb + eps

        na = aa / denom
        nb = bb / denom

        dot = (
            na * nb
        ).sum(dim=-1)

        norm_a = torch.sqrt(
            (
                na * na
            ).sum(dim=-1)
            + eps
        )

        norm_b = torch.sqrt(
            (
                nb * nb
            ).sum(dim=-1)
            + eps
        )

        cosine = (
            dot
            / (
                norm_a
                * norm_b
            )
        )

        result[start:end] = (
            1.0 - cosine
        ).detach().cpu().numpy()

    return result


def evaluate_task2vec():
    print("\n" + "=" * 80)
    print("TASK2VEC")
    print("=" * 80)

    train = np.load(
        os.path.join(
            BASE,
            "task2vec_train.npz",
        )
    )

    val = np.load(
        os.path.join(
            BASE,
            "task2vec_val.npz",
        )
    )

    x_train = train["embeddings"]
    y_train = train["domains"]

    x_val = val["embeddings"]
    y_val = val["domains"]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Distance matrix hesaplanıyor...")

    D_train = task2vec_distance_matrix(
        x_train,
        x_train,
        device,
    )

    np.fill_diagonal(
        D_train,
        0.0,
    )

    # --------------------------------------
    # K=9 clustering
    # --------------------------------------

    clusterer = AgglomerativeClustering(
        n_clusters=9,
        metric="precomputed",
        linkage="average",
    )

    clusters = clusterer.fit_predict(
        D_train
    )

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

    # --------------------------------------
    # Her domain için medoid
    # --------------------------------------

    medoids = []

    for d in DOMAINS:
        idx = np.where(
            y_train == d
        )[0]

        sub = D_train[
            np.ix_(idx, idx)
        ]

        mean_dist = (
            sub.mean(axis=1)
        )

        medoid_local = (
            mean_dist.argmin()
        )

        medoid_global = (
            idx[medoid_local]
        )

        medoids.append(
            x_train[medoid_global]
        )

    medoids = np.stack(medoids)

    # --------------------------------------
    # Validation routing
    # --------------------------------------

    D_val = task2vec_distance_matrix(
        x_val,
        medoids,
        device,
    )

    pred_idx = D_val.argmin(axis=1)

    y_pred = np.array(
        [DOMAINS[i] for i in pred_idx]
    )

    acc9 = np.mean(
        y_pred == y_val
    )

    acc3 = family_accuracy(
        y_val,
        y_pred,
    )

    print(f"Purity : {purity:.4f}")
    print(f"NMI    : {nmi:.4f}")
    print(f"ARI    : {ari:.4f}")

    print()
    print(f"Acc9   : {acc9:.4f}")
    print(f"Acc3   : {acc3:.4f}")

    print_domain_accuracy(
        y_val,
        y_pred,
    )

    print_family_accuracy(
        y_val,
        y_pred,
    )

    print_confusion(
        y_val,
        y_pred,
    )


if __name__ == "__main__":
    evaluate_frozen_mean()
    evaluate_task2vec()