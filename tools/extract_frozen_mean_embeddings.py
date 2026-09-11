import argparse
import os

import numpy as np
import torch

from models import encoders


def load_encoder(ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    encoder = encoders.load(ckpt)
    encoder = encoder.to(device)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


@torch.no_grad()
def extract_split(stream_path, encoder, device, output_path):
    payload = torch.load(
        stream_path,
        map_location="cpu",
        weights_only=False,
    )

    embeddings = []
    domains = []
    task_ids = []

    tasks = payload["tasks"]

    for i, task in enumerate(tasks):
        x_shot = task["x_shot"].to(device)

        # 25 support image -> ResNet18 feature
        try:
            feat = encoder(x_shot, None, 0)
        except TypeError:
            try:
                feat = encoder(x_shot, None)
            except TypeError:
                feat = encoder(x_shot)

        if feat.dim() > 2:
            feat = feat.flatten(1)

        # [25, 512] -> [512]
        task_embedding = feat.mean(dim=0)

        embeddings.append(
            task_embedding.detach().cpu().numpy()
        )
        domains.append(task["domain"])
        task_ids.append(task["task_id"])

        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(tasks)} task")

    embeddings = np.stack(embeddings)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    np.savez(
        output_path,
        embeddings=embeddings,
        domains=np.array(domains),
        task_ids=np.array(task_ids),
    )

    print("\nKaydedildi:", output_path)
    print("Embedding shape:", embeddings.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoder-ckpt",
        required=True,
    )

    parser.add_argument(
        "--stream-dir",
        default="materials/embedding_task_stream_5shot",
    )

    parser.add_argument(
        "--output-dir",
        default="materials/embedding_comparison_5shot",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    encoder = load_encoder(
        args.encoder_ckpt,
        device,
    )

    extract_split(
        os.path.join(
            args.stream_dir,
            "meta_train_800.pt"
        ),
        encoder,
        device,
        os.path.join(
            args.output_dir,
            "frozen_mean_train.npz"
        ),
    )

    extract_split(
        os.path.join(
            args.stream_dir,
            "meta_val_600.pt"
        ),
        encoder,
        device,
        os.path.join(
            args.output_dir,
            "frozen_mean_val.npz"
        ),
    )


if __name__ == "__main__":
    main()