import os
import random
import argparse

import numpy as np
import torch

import datasets


DOMAINS = [
    "ACT_40",
    "ACT_410",
    "SPT",
    "RESISC",
    "RSICB",
    "RSD",
    "TEX",
    "TEX_ALOT",
    "TEX_DTD",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def export_split(
    split,
    n_batch,
    n_episode,
    output_path,
    root_path,
    manifest,
    seed,
):
    set_seed(seed)

    dataset = datasets.make(
        "meta-album",
        root_path=root_path,
        split=split,
        domains=DOMAINS,
        class_split_manifest=manifest,
        image_size=128,
        normalization=False,
        transform=None,
        n_batch=n_batch,
        n_episode=n_episode,
        n_way=5,
        n_shot=5,
        n_query=15,
    )

    # Dataset normalde domain bilgisini döndürmüyor.
    # Seçilen domain indeksini burada yakalıyoruz.
    total_tasks = len(dataset)
    n_domains = len(dataset.domains)

    base_count = total_tasks // n_domains
    remainder = total_tasks % n_domains

    domain_schedule = []

    for domain_idx in range(n_domains):
        count = base_count + (1 if domain_idx < remainder else 0)
        domain_schedule.extend([domain_idx] * count)

    rng = np.random.RandomState(seed)
    rng.shuffle(domain_schedule)

    schedule_pos = 0

    def balanced_sample_domain_idx():
        nonlocal schedule_pos

        idx = domain_schedule[schedule_pos]
        schedule_pos += 1

        dataset._last_domain_idx = idx
        return idx

    dataset._sample_domain_idx = balanced_sample_domain_idx

    tasks = []

    for task_id in range(len(dataset)):
        x_shot, x_query, y_shot, y_query = dataset[task_id]

        domain_idx = dataset._last_domain_idx
        domain = dataset.domains[domain_idx]

        tasks.append({
            "task_id": task_id,
            "domain": domain,

            # İki embedding yöntemi de SADECE bunları kullanacak.
            "x_shot": x_shot.cpu(),
            "y_shot": y_shot.cpu(),

            # Şimdilik karşılaştırmada kullanılmayacak.
            # Kontrol / ileride gerekli olursa elimizde olsun.
            "x_query": x_query.cpu(),
            "y_query": y_query.cpu(),
        })

        if (task_id + 1) % 100 == 0:
            print(
                f"{split}: {task_id + 1}/{len(dataset)} task"
            )

    payload = {
        "split": split,
        "seed": seed,
        "domains": DOMAINS,
        "n_way": 5,
        "n_shot": 5,
        "n_query": 15,
        "image_size": 128,
        "normalization": False,
        "manifest": manifest,
        "tasks": tasks,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)

    print()
    print("Kaydedildi:", output_path)
    print("Toplam task:", len(tasks))

    domain_counts = {}
    for task in tasks:
        d = task["domain"]
        domain_counts[d] = domain_counts.get(d, 0) + 1

    print("\nDomain dağılımı:")
    for domain in DOMAINS:
        print(
            f"{domain:10s}: {domain_counts.get(domain, 0)}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="materials/album_128",
    )
    parser.add_argument(
        "--manifest",
        default="materials/album_128/"
                "geometry_cluster_abc_class_split.json",
    )
    parser.add_argument(
        "--output-dir",
        default="materials/embedding_task_stream_5shot",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    export_split(
        split="meta-train",
        n_batch=200,
        n_episode=4,
        output_path=os.path.join(
            args.output_dir,
            "meta_train_800.pt",
        ),
        root_path=args.root,
        manifest=args.manifest,
        seed=args.seed,
    )

    export_split(
        split="meta-val",
        n_batch=100,
        n_episode=6,
        output_path=os.path.join(
            args.output_dir,
            "meta_val_600.pt",
        ),
        root_path=args.root,
        manifest=args.manifest,
        seed=args.seed + 1,
    )


if __name__ == "__main__":
    main()