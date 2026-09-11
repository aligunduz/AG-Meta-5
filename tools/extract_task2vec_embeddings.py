import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import encoders


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_forward(encoder, x):
    """
    AG-Meta encoder API'sindeki farklı forward imzalarına uyum sağlar.
    """
    try:
        return encoder(x, None, 0)
    except TypeError:
        try:
            return encoder(x, None)
        except TypeError:
            return encoder(x)


def load_encoder(ckpt_path, device):
    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    encoder = encoders.load(ckpt)
    encoder = encoder.to(device)
    encoder.eval()

    return encoder


@torch.no_grad()
def extract_features(encoder, x):
    feat = encoder_forward(encoder, x)

    if feat.dim() > 2:
        feat = feat.flatten(1)

    return feat.float()


def fit_task_classifier(
    features,
    labels,
    n_way,
    steps=100,
    lr=0.05,
    weight_decay=1e-4,
):
    """
    Task2Vec probe classifier.

    Encoder sabit kalır.
    Sadece support feature'ları üzerinde 5-way lineer classifier fit edilir.
    """

    classifier = nn.Linear(
        features.size(1),
        n_way,
        bias=True,
    ).to(features.device)

    # Task'ler arasında initialization farkı oluşmasın.
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    classifier.train()

    for _ in range(steps):
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    classifier.eval()

    with torch.no_grad():
        pred = classifier(features).argmax(dim=1)
        acc = (pred == labels).float().mean().item()

    # Fisher hesaplarken classifier güncellenmeyecek.
    for p in classifier.parameters():
        p.requires_grad_(False)

    return classifier, acc


def get_fisher_parameters(encoder):
    """
    Task2Vec embeddinginde filter-wise Fisher kullanıyoruz.

    Bias ve 1D BN parametrelerini dışarıda bırakıyoruz.
    Conv / Linear weight tensorleri kullanılıyor.
    """
    params = []

    for name, param in encoder.named_parameters():
        if param.dim() >= 2:
            params.append((name, param))

    return params


def compute_task2vec_embedding(
    encoder,
    classifier,
    x_shot,
    mc_samples=4,
):
    """
    Monte-Carlo diagonal Fisher approximation.

    Her encoder weight tensoru için:
        gradient^2

    hesaplanır.

    Daha sonra her output filter/neuron içerisindeki Fisher değerleri
    ortalanarak filter-wise Task2Vec embedding elde edilir.
    """

    encoder.eval()

    fisher_params = get_fisher_parameters(encoder)

    # Fisher hesaplamak için encoder parametrelerinde gradient gerekli,
    # fakat optimizer step YOK. Encoder değişmeyecek.
    original_requires_grad = {}

    for name, p in encoder.named_parameters():
        original_requires_grad[name] = p.requires_grad
        p.requires_grad_(True)

    layer_fisher = [
        torch.zeros(
            p.shape[0],
            device=x_shot.device,
            dtype=torch.float32,
        )
        for _, p in fisher_params
    ]

    n = x_shot.size(0)

    for _ in range(mc_samples):

        encoder.zero_grad(set_to_none=True)

        feat = encoder_forward(encoder, x_shot)

        if feat.dim() > 2:
            feat = feat.flatten(1)

        feat = feat.float()

        logits = classifier(feat)

        # Monte-Carlo Fisher:
        # y ~ p(y|x)
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)
            sampled_y = torch.multinomial(
                probs,
                num_samples=1,
            ).squeeze(1)

        log_probs = F.log_softmax(logits, dim=1)

        selected_log_prob = log_probs[
            torch.arange(n, device=x_shot.device),
            sampled_y,
        ]

        # 1/sqrt(N) ölçeklemesi:
        # squared gradient expectation'ını yaklaşık sample-average
        # Fisher ölçeğinde tutar.
        score = selected_log_prob.sum() / np.sqrt(n)

        grads = torch.autograd.grad(
            score,
            [p for _, p in fisher_params],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        for i, grad in enumerate(grads):
            if grad is None:
                continue

            g2 = grad.float().pow(2)

            # Örn:
            # Conv: [out, in, k, k] -> [out]
            # Linear: [out, in]      -> [out]
            reduce_dims = tuple(range(1, g2.dim()))

            filter_fisher = g2.mean(dim=reduce_dims)

            layer_fisher[i] += filter_fisher

    layer_fisher = [
        f / float(mc_samples)
        for f in layer_fisher
    ]

    embedding = torch.cat(layer_fisher, dim=0)

    # requires_grad durumunu geri yükle.
    for name, p in encoder.named_parameters():
        p.requires_grad_(original_requires_grad[name])

    encoder.zero_grad(set_to_none=True)

    return embedding.detach()


def extract_split(
    stream_path,
    encoder,
    device,
    output_path,
    classifier_steps,
    classifier_lr,
    mc_samples,
    seed,
):
    payload = torch.load(
        stream_path,
        map_location="cpu",
        weights_only=False,
    )

    embeddings = []
    domains = []
    task_ids = []
    probe_accuracies = []

    tasks = payload["tasks"]

    for i, task in enumerate(tasks):

        # Her task deterministik olsun.
        task_seed = seed + int(task["task_id"])
        set_seed(task_seed)

        x_shot = task["x_shot"].to(
            device,
            non_blocking=True,
        )

        y_shot = task["y_shot"].to(
            device,
            non_blocking=True,
        ).long()

        # Emin olmak için label'ları 0,...,4'e remap et.
        _, y_shot = torch.unique(
            y_shot,
            sorted=True,
            return_inverse=True,
        )

        # --------------------------------------------------
        # 1. Frozen ResNet feature
        # --------------------------------------------------
        with torch.no_grad():
            features = extract_features(
                encoder,
                x_shot,
            )

        # --------------------------------------------------
        # 2. Task-specific 5-way classifier
        # --------------------------------------------------
        classifier, probe_acc = fit_task_classifier(
            features,
            y_shot,
            n_way=5,
            steps=classifier_steps,
            lr=classifier_lr,
        )

        # --------------------------------------------------
        # 3. Monte-Carlo Fisher -> Task2Vec
        # --------------------------------------------------
        embedding = compute_task2vec_embedding(
            encoder,
            classifier,
            x_shot,
            mc_samples=mc_samples,
        )

        embeddings.append(
            embedding.cpu().numpy()
        )

        domains.append(task["domain"])
        task_ids.append(task["task_id"])
        probe_accuracies.append(probe_acc)

        if (i + 1) % 50 == 0:
            current_mean_acc = np.mean(
                probe_accuracies[-50:]
            )

            print(
                f"{i + 1}/{len(tasks)} task | "
                f"probe_acc(last50)="
                f"{current_mean_acc:.4f}"
            )

    embeddings = np.stack(embeddings)

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True,
    )

    np.savez(
        output_path,
        embeddings=embeddings,
        domains=np.array(domains),
        task_ids=np.array(task_ids),
        probe_accuracies=np.array(
            probe_accuracies,
            dtype=np.float32,
        ),
    )

    print()
    print("Kaydedildi:", output_path)
    print("Embedding shape:", embeddings.shape)
    print(
        "Probe accuracy:",
        f"{np.mean(probe_accuracies):.4f}"
    )

    print(
        "Embedding min/max:",
        float(embeddings.min()),
        float(embeddings.max()),
    )


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

    parser.add_argument(
        "--classifier-steps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--mc-samples",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    encoder = load_encoder(
        args.encoder_ckpt,
        device,
    )

    fisher_params = get_fisher_parameters(
        encoder
    )

    embedding_dim = sum(
        p.shape[0]
        for _, p in fisher_params
    )

    print(
        "Task2Vec Fisher layers:",
        len(fisher_params),
    )

    print(
        "Task2Vec embedding dimension:",
        embedding_dim,
    )

    # -----------------------------
    # TRAIN
    # -----------------------------

    extract_split(
        stream_path=os.path.join(
            args.stream_dir,
            "meta_train_800.pt",
        ),
        encoder=encoder,
        device=device,
        output_path=os.path.join(
            args.output_dir,
            "task2vec_train.npz",
        ),
        classifier_steps=args.classifier_steps,
        classifier_lr=args.classifier_lr,
        mc_samples=args.mc_samples,
        seed=args.seed,
    )

    # -----------------------------
    # VAL
    # -----------------------------

    extract_split(
        stream_path=os.path.join(
            args.stream_dir,
            "meta_val_600.pt",
        ),
        encoder=encoder,
        device=device,
        output_path=os.path.join(
            args.output_dir,
            "task2vec_val.npz",
        ),
        classifier_steps=args.classifier_steps,
        classifier_lr=args.classifier_lr,
        mc_samples=args.mc_samples,
        seed=args.seed + 100000,
    )


if __name__ == "__main__":
    main()