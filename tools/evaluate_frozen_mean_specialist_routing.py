import os
import numpy as np
import pandas as pd

BASE = "materials/embedding_comparison_5shot"

TRAIN_NPZ = os.path.join(BASE, "frozen_mean_train.npz")
VAL_NPZ = os.path.join(BASE, "frozen_mean_val.npz")
RESULT_CSV = os.path.join(BASE, "specialist_oracle_vs_universal.csv")

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

ANCHORS = ["A", "B", "C"]


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def paired_stats(a, b):
    """
    a - b paired difference mean and standard error.
    """
    d = np.asarray(a) - np.asarray(b)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return mean, se


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

train = np.load(TRAIN_NPZ, allow_pickle=True)
val = np.load(VAL_NPZ, allow_pickle=True)

x_train = train["embeddings"].astype(np.float64)
y_train = train["domains"].astype(str)

x_val = val["embeddings"].astype(np.float64)
y_val = val["domains"].astype(str)

x_train = l2_normalize(x_train)
x_val = l2_normalize(x_val)

print("Train:", x_train.shape)
print("Val  :", x_val.shape)


# --------------------------------------------------
# A/B/C family centers from TRAIN embeddings
# --------------------------------------------------

centers = []

for family in ANCHORS:
    mask = np.array([
        FAMILIES[d] == family
        for d in y_train
    ])

    center = x_train[mask].mean(axis=0)
    center = center / (np.linalg.norm(center) + 1e-12)

    centers.append(center)

    print(
        f"Family {family} center: "
        f"{mask.sum()} train tasks"
    )

centers = np.stack(centers)


# --------------------------------------------------
# Route every validation task
# cosine similarity => largest = closest
# --------------------------------------------------

similarity = x_val @ centers.T

route_idx = similarity.argmax(axis=1)

routes = np.array([
    ANCHORS[i]
    for i in route_idx
])


# --------------------------------------------------
# Load per-task specialist accuracies
# --------------------------------------------------

df = pd.read_csv(RESULT_CSV)

if len(df) != len(x_val):
    raise RuntimeError(
        f"Length mismatch: CSV={len(df)}, val embeddings={len(x_val)}"
    )

# Stream order sanity check
if not np.array_equal(
    df["domain"].astype(str).to_numpy(),
    y_val
):
    raise RuntimeError(
        "CSV domain order != embedding domain order. "
        "Task alignment güvenli değil."
    )

print("\nTask alignment: OK")


# --------------------------------------------------
# Select actually achieved accuracy according to route
# --------------------------------------------------

acc_embed = np.zeros(len(df), dtype=np.float64)

for i, route in enumerate(routes):
    acc_embed[i] = df.loc[i, f"acc_{route}"]

df["embedding_route"] = routes
df["acc_embedding_routed"] = acc_embed

df["correct_family_route"] = [
    int(route == FAMILIES[domain])
    for route, domain in zip(
        routes,
        df["domain"],
    )
]


# --------------------------------------------------
# Overall summary
# --------------------------------------------------

mean_embed = df["acc_embedding_routed"].mean()
mean_u = df["acc_U"].mean()
mean_oracle = df["oracle_accuracy"].mean()
mean_b = df["acc_B"].mean()

diff_u, se_u = paired_stats(
    df["acc_embedding_routed"],
    df["acc_U"],
)

diff_oracle, se_oracle = paired_stats(
    df["acc_embedding_routed"],
    df["oracle_accuracy"],
)

diff_b, se_b = paired_stats(
    df["acc_embedding_routed"],
    df["acc_B"],
)

print("\n" + "=" * 78)
print("OVERALL")
print("=" * 78)

print(f"Embedding-routed      : {mean_embed:.4f}")
print(f"Universal             : {mean_u:.4f}")
print(f"Oracle specialist     : {mean_oracle:.4f}")
print(f"Fixed specialist B    : {mean_b:.4f}")

print()
print(
    f"Embed - Universal     : "
    f"{diff_u:+.4f} ± {se_u:.4f}"
)

print(
    f"Embed - Oracle        : "
    f"{diff_oracle:+.4f} ± {se_oracle:.4f}"
)

print(
    f"Embed - Fixed B       : "
    f"{diff_b:+.4f} ± {se_b:.4f}"
)

print()
print(
    f"Family routing Acc3   : "
    f"{df['correct_family_route'].mean():.4f}"
)


# --------------------------------------------------
# How often each anchor is selected
# --------------------------------------------------

print("\nRouting distribution:")

for anchor in ANCHORS:
    n = (df["embedding_route"] == anchor).sum()
    print(
        f"  {anchor}: "
        f"{n:3d}/{len(df)} "
        f"({n / len(df):.4f})"
    )


# --------------------------------------------------
# Domain-level results
# --------------------------------------------------

print("\n" + "=" * 78)
print("BY DOMAIN")
print("=" * 78)

domains = [
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

domain_rows = []

for domain in domains:

    sub = df[
        df["domain"] == domain
    ].copy()

    e = sub["acc_embedding_routed"].mean()
    u = sub["acc_U"].mean()
    o = sub["oracle_accuracy"].mean()
    b = sub["acc_B"].mean()

    diff, se = paired_stats(
        sub["acc_embedding_routed"],
        sub["acc_U"],
    )

    route_counts = {
        a: (
            sub["embedding_route"] == a
        ).mean()
        for a in ANCHORS
    }

    domain_rows.append({
        "domain": domain,
        "n": len(sub),

        "route_A": route_counts["A"],
        "route_B": route_counts["B"],
        "route_C": route_counts["C"],

        "embedding_acc": e,
        "universal_acc": u,
        "oracle_acc": o,
        "fixed_B_acc": b,

        "embed_minus_U": diff,
        "embed_minus_U_se": se,
    })

    print(f"\n{domain} | n={len(sub)}")

    print(
        "  route: "
        f"A={route_counts['A']:.3f} "
        f"B={route_counts['B']:.3f} "
        f"C={route_counts['C']:.3f}"
    )

    print(f"  Embedding routed : {e:.4f}")
    print(f"  Universal        : {u:.4f}")
    print(f"  Oracle specialist: {o:.4f}")

    print(
        f"  Embed - U        : "
        f"{diff:+.4f} ± {se:.4f}"
    )


# --------------------------------------------------
# Save outputs
# --------------------------------------------------

domain_df = pd.DataFrame(domain_rows)

output_tasks = os.path.join(
    BASE,
    "frozen_mean_specialist_routing_tasks.csv",
)

output_domains = os.path.join(
    BASE,
    "frozen_mean_specialist_routing_domains.csv",
)

df.to_csv(
    output_tasks,
    index=False,
)

domain_df.to_csv(
    output_domains,
    index=False,
)

print("\nSaved:")
print(output_tasks)
print(output_domains)