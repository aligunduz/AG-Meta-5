"""Frozen support encoder + fixed PCA-whitening + top-k soft routing."""

import math

import numpy as np
import torch
import torch.nn as nn

from . import encoders


class SoftRouter(nn.Module):
    """
    Input:
        x_shot: [n_episode, n_support, C, H, W]
                or [n_support, C, H, W]

    Output dictionary:
        weights:        [n_episode, n_anchor]
        indices:        [n_episode, top_k]
        selected_weights: [n_episode, top_k]
        distances:      [n_episode, n_anchor]
        embeddings:     [n_episode, original_dim]
        task_coordinates: [n_episode, pca_dim]

    Even a single task produces outputs with an episode dimension.
    This module never updates its encoder, geometry, or anchor parameters.
    """

    def __init__(
        self,
        encoder_ckpt,
        geometry_path,
        top_k=2,
        tau=0.5,
    ):
        super().__init__()

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise ValueError("top_k must be an integer.")
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and positive.")

        # Same encoder loader used by the embedding extraction script.
        checkpoint = torch.load(
            encoder_ckpt,
            map_location="cpu",
            weights_only=False,
        )
        self.encoder = encoders.load(checkpoint).float()
        self.encoder.requires_grad_(False)
        self.encoder.eval()

        with np.load(geometry_path, allow_pickle=False) as data:
            geometry = {
                key: np.array(data[key], dtype=np.float64, copy=True)
                for key in [
                    "mean",
                    "components",
                    "whitening_scale",
                    "centers",
                ]
            }
            if "metric" in data:
                if str(data["metric"].item()) != "euclidean":
                    raise ValueError("Only Euclidean geometry is supported.")

        mean = geometry["mean"]
        components = geometry["components"]
        scale = geometry["whitening_scale"]
        centers = geometry["centers"]

        if mean.ndim != 1 or components.ndim != 2:
            raise ValueError("Invalid PCA mean/components dimensions.")
        if components.shape[1] != mean.size:
            raise ValueError("PCA input dimensions do not match.")
        if scale.shape != (components.shape[0],):
            raise ValueError("Invalid whitening_scale dimensions.")
        if centers.ndim != 2 or centers.shape[1] != scale.size:
            raise ValueError("Anchor centers do not match the PCA space.")
        if not all(np.isfinite(v).all() for v in geometry.values()):
            raise ValueError("Geometry contains NaN/Inf.")
        if np.any(scale <= 0):
            raise ValueError("Whitening scales must be positive.")
        if not 1 <= top_k <= centers.shape[0]:
            raise ValueError("top_k must be between 1 and anchor count.")

        # Buffers move with .to(device) and are included in state_dict.
        # Float64 retains the precision of the offline geometry analysis.
        self.register_buffer("pca_mean", torch.from_numpy(mean))
        self.register_buffer("pca_components", torch.from_numpy(components))
        self.register_buffer("whitening_scale", torch.from_numpy(scale))
        self.register_buffer("centers", torch.from_numpy(centers))

        self.top_k = top_k
        self.tau = float(tau)
        self.train(False)

    @property
    def n_anchors(self):
        return self.centers.shape[0]

    def train(self, mode=True):
        # Parent model.train() must not switch the frozen router to train.
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x_shot):
        if x_shot.ndim == 4:
            x_shot = x_shot.unsqueeze(0)

        if x_shot.ndim != 5:
            raise ValueError(
                "Expected [episodes, support, C, H, W] "
                "or [support, C, H, W]."
            )
        if x_shot.shape[0] == 0 or x_shot.shape[1] == 0:
            raise ValueError("Episode and support counts must be positive.")

        device = self.pca_mean.device
        if x_shot.device != device:
            raise ValueError(
                "Router and support must be on the same device. "
                "Move the router with router.to(x_shot.device)."
            )
        if next(self.encoder.parameters()).dtype != torch.float32:
            raise ValueError(
                "Keep the router encoder in float32; use AMP for "
                "the meta-model instead of router.half()/bfloat16()."
            )

        self.encoder.eval()

        # Match offline extraction: FP32 encoder, one task at a time.
        # This matters because BatchNorm has track_running_stats=False.
        with torch.autocast(device_type=device.type, enabled=False):
            embeddings = []

            for support in x_shot:
                features = self.encoder(support.float(), None, 0)

                if features.ndim > 2:
                    features = features.flatten(1)

                if (
                    features.ndim != 2
                    or features.shape[0] != support.shape[0]
                    or features.shape[1] != self.pca_mean.numel()
                ):
                    raise ValueError("Unexpected encoder feature shape.")

                embeddings.append(features.mean(dim=0))

            embeddings = torch.stack(embeddings)
            if not torch.isfinite(embeddings).all():
                raise ValueError("Encoder embeddings contain NaN/Inf.")

            coordinates = (
                (embeddings.to(self.pca_mean.dtype) - self.pca_mean)
                @ self.pca_components.T
            ) / self.whitening_scale

            distances = torch.linalg.vector_norm(
                coordinates[:, None, :] - self.centers[None, :, :],
                dim=-1,
            )
            if not torch.isfinite(distances).all():
                raise ValueError("Routing distances contain NaN/Inf.")

            # Stable ordering matches the offline NumPy tie-breaking rule.
            indices = torch.argsort(
                distances, dim=1, stable=True
            )[:, :self.top_k]

            selected_distances = distances.gather(1, indices)
            logits = -(
                selected_distances - selected_distances[:, :1]
            ) / self.tau

            selected_weights = torch.softmax(logits, dim=1).float()
            weights = torch.zeros(
                (x_shot.shape[0], self.n_anchors),
                device=device,
                dtype=torch.float32,
            )
            weights.scatter_(1, indices, selected_weights)

        return {
            "weights": weights,
            "indices": indices,
            "selected_weights": selected_weights,
            "distances": distances,
            "embeddings": embeddings,
            "task_coordinates": coordinates,
        }