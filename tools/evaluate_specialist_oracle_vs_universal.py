import argparse
import os
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import amp
from tqdm import tqdm

import models
import utils


FAMILIES = {
    "ACT_40": "A",
    "ACT_410": "A",
    "SPT": "A",

    "RESISC": "B",
    "RSICB": "B",
    "RSD": "B",

    "TEX": "C",
    "TEX_ALOT": "C",
    "TEX_DTD": "C",
}


def load_model(ckpt_path, device, inner_args):
    print(f"\nLoading: {ckpt_path}")

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    # reset_classifier=False olduğu için checkpoint classifier'ını da yükle.
    model = models.load(
        ckpt,
        load_clf=(not inner_args["reset_classifier"]),
    )

    model = model.to(device)
    model.eval()

    epoch = ckpt.get("training", {}).get("epoch", None)
    max_va = ckpt.get("training", {}).get("max_va", None)

    print(f"  checkpoint epoch: {epoch}")
    print(f"  checkpoint max_va: {max_va}")

    return model


def evaluate_model(
    model,
    tasks,
    inner_args,
    device,
    use_amp=True,
    batch_size=4,
):
    """
    Aynı sabit task listesi üzerinde bir modeli değerlendirir.
    Her task için query accuracy ve loss döndürür.
    """

    accuracies = np.zeros(len(tasks), dtype=np.float32)
    losses = np.zeros(len(tasks), dtype=np.float32)

    for start in tqdm(
        range(0, len(tasks), batch_size),
        desc="evaluating",
    ):
        batch = tasks[start:start + batch_size]

        x_shot = torch.stack([
            task["x_shot"] for task in batch
        ]).to(device, non_blocking=True)

        y_shot = torch.stack([
            task["y_shot"] for task in batch
        ]).to(device, non_blocking=True).long()

        x_query = torch.stack([
            task["x_query"] for task in batch
        ]).to(device, non_blocking=True)

        y_query = torch.stack([
            task["y_query"] for task in batch
        ]).to(device, non_blocking=True).long()

        if inner_args["reset_classifier"]:
            model.reset_classifier()

        with torch.no_grad(), amp.autocast(
            "cuda",
            enabled=(use_amp and device.type == "cuda"),
            dtype=torch.bfloat16,
        ):
            logits = model(
                x_shot,
                x_query,
                y_shot,
                inner_args,
                meta_train=False,
                use_gradient_transport=False,
            )

        # logits: [n_episode, n_query_total, n_way]
        for j in range(len(batch)):
            logits_j = logits[j].float()
            labels_j = y_query[j]

            pred_j = logits_j.argmax(dim=-1)

            acc_j = (
                pred_j == labels_j
            ).float().mean().item()

            loss_j = F.cross_entropy(
                logits_j,
                labels_j,
            ).item()

            idx = start + j

            accuracies[idx] = acc_j
            losses[idx] = loss_j

    return accuracies, losses


def evaluate_checkpoint(
    name,
    path,
    tasks,
    inner_args,
    device,
    use_amp,
):
    model = load_model(
        path,
        device,
        inner_args,
    )

    acc, loss = evaluate_model(
        model,
        tasks,
        inner_args,
        device,
        use_amp=use_amp,
        batch_size=4,
    )

    print(
        f"{name}: "
        f"mean_acc={acc.mean():.4f} "
        f"mean_loss={loss.mean():.4f}"
    )

    del model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return acc, loss


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stream",
        default=(
            "materials/"
            "embedding_task_stream_5shot/"
            "meta_val_600.pt"
        ),
    )

    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--model-c", required=True)
    parser.add_argument("--model-u", required=True)

    parser.add_argument(
        "--output",
        default=(
            "materials/"
            "embedding_comparison_5shot/"
            "specialist_oracle_vs_universal.csv"
        ),
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    payload = torch.load(
        args.stream,
        map_location="cpu",
        weights_only=False,
    )

    tasks = payload["tasks"]

    print("Tasks:", len(tasks))
    print(
        "Protocol:",
        payload["n_way"],
        "way",
        payload["n_shot"],
        "shot",
        payload["n_query"],
        "query",
    )

    # Önceki paired specialist protokolümüzle aynı.
    inner_args = utils.config_inner_args({
        "n_step": 5,
        "encoder_lr": 0.01,
        "classifier_lr": 0.01,
        "first_order": True,
        "momentum": 0.0,
        "weight_decay": 0.0,
        "reset_classifier": False,
        "frozen": ["bn"],
    })

    use_amp = not args.no_amp

    # --------------------------------------------------
    # Aynı 600 taskı dört modele sırayla veriyoruz.
    # Böylece aynı anda 4 ResNet GPU'da tutulmuyor.
    # --------------------------------------------------

    acc_A, loss_A = evaluate_checkpoint(
        "A",
        args.model_a,
        tasks,
        inner_args,
        device,
        use_amp,
    )

    acc_B, loss_B = evaluate_checkpoint(
        "B",
        args.model_b,
        tasks,
        inner_args,
        device,
        use_amp,
    )

    acc_C, loss_C = evaluate_checkpoint(
        "C",
        args.model_c,
        tasks,
        inner_args,
        device,
        use_amp,
    )

    acc_U, loss_U = evaluate_checkpoint(
        "U",
        args.model_u,
        tasks,
        inner_args,
        device,
        use_amp,
    )

    # --------------------------------------------------
    # Per-task tablo
    # --------------------------------------------------

    rows = []

    specialists = np.stack(
        [acc_A, acc_B, acc_C],
        axis=1,
    )

    specialist_names = np.array(
        ["A", "B", "C"]
    )

    eps = 1e-8

    for i, task in enumerate(tasks):

        scores = specialists[i]

        order = np.argsort(-scores)

        best_value = scores[order[0]]
        second_value = scores[order[1]]

        winners = specialist_names[
            np.abs(scores - best_value) <= eps
        ]

        oracle_anchor = "|".join(winners.tolist())

        oracle_margin = (
            best_value - second_value
        )

        universal_margin = (
            acc_U[i] - best_value
        )

        universal_better = (
            acc_U[i] > best_value + eps
        )

        oracle_better = (
            best_value > acc_U[i] + eps
        )

        tie = (
            abs(acc_U[i] - best_value)
            <= eps
        )

        domain = task["domain"]
        family = FAMILIES[domain]

        rows.append({
            "task_id": int(task["task_id"]),
            "domain": domain,
            "family_label": family,

            "acc_A": float(acc_A[i]),
            "acc_B": float(acc_B[i]),
            "acc_C": float(acc_C[i]),
            "acc_U": float(acc_U[i]),

            "loss_A": float(loss_A[i]),
            "loss_B": float(loss_B[i]),
            "loss_C": float(loss_C[i]),
            "loss_U": float(loss_U[i]),

            "oracle_anchor": oracle_anchor,
            "oracle_n_ties": len(winners),

            "oracle_accuracy": float(best_value),

            "second_best_anchor":
                specialist_names[order[1]],

            "second_best_accuracy":
                float(second_value),

            "oracle_margin":
                float(oracle_margin),

            "universal_better":
                int(universal_better),

            "oracle_specialist_better":
                int(oracle_better),

            "universal_oracle_tie":
                int(tie),

            "universal_margin":
                float(universal_margin),
        })

    df = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(args.output),
        exist_ok=True,
    )

    df.to_csv(
        args.output,
        index=False,
    )

    # --------------------------------------------------
    # Özet
    # --------------------------------------------------

    oracle = df["oracle_accuracy"].to_numpy()
    universal = df["acc_U"].to_numpy()

    print()
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    print(
        f"Mean A              : "
        f"{df['acc_A'].mean():.4f}"
    )

    print(
        f"Mean B              : "
        f"{df['acc_B'].mean():.4f}"
    )

    print(
        f"Mean C              : "
        f"{df['acc_C'].mean():.4f}"
    )

    print(
        f"Mean Universal      : "
        f"{universal.mean():.4f}"
    )

    print(
        f"Mean Oracle Spec.   : "
        f"{oracle.mean():.4f}"
    )

    print(
        f"Oracle - Universal  : "
        f"{(oracle - universal).mean():+.4f}"
    )

    print()
    print(
        "Oracle > Universal : "
        f"{df['oracle_specialist_better'].mean():.4f}"
    )

    print(
        "Universal > Oracle : "
        f"{df['universal_better'].mean():.4f}"
    )

    print(
        "Oracle = Universal : "
        f"{df['universal_oracle_tie'].mean():.4f}"
    )

    # En iyi sabit specialist
    fixed_means = {
        "A": df["acc_A"].mean(),
        "B": df["acc_B"].mean(),
        "C": df["acc_C"].mean(),
    }

    best_fixed_name = max(
        fixed_means,
        key=fixed_means.get,
    )

    best_fixed_acc = fixed_means[
        best_fixed_name
    ]

    print()
    print(
        f"Best fixed specialist: "
        f"{best_fixed_name} "
        f"({best_fixed_acc:.4f})"
    )

    print(
        f"Oracle specialist gain "
        f"over best fixed: "
        f"{oracle.mean() - best_fixed_acc:+.4f}"
    )

    # --------------------------------------------------
    # Family bazında
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("BY FAMILY")
    print("=" * 70)

    for family in ["A", "B", "C"]:

        sub = df[
            df["family_label"] == family
        ]

        print(
            f"\nFamily {family} | "
            f"n={len(sub)}"
        )

        print(
            f"  Universal : "
            f"{sub['acc_U'].mean():.4f}"
        )

        print(
            f"  Oracle    : "
            f"{sub['oracle_accuracy'].mean():.4f}"
        )

        print(
            f"  Oracle-U  : "
            f"{(
                sub['oracle_accuracy']
                - sub['acc_U']
            ).mean():+.4f}"
        )

        print(
            f"  O > U rate: "
            f"{sub['oracle_specialist_better'].mean():.4f}"
        )

        print(
            f"  U > O rate: "
            f"{sub['universal_better'].mean():.4f}"
        )

    print()
    print("Kaydedildi:")
    print(args.output)


if __name__ == "__main__":
    main()