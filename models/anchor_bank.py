from collections import OrderedDict

import torch
import torch.nn as nn


class AnchorBank(nn.Module):
    """Independent encoder/classifier parameter copies from one root."""

    def __init__(self, root_model, n_anchors):
        super().__init__()

        if n_anchors < 1:
            raise ValueError("n_anchors must be positive.")

        # Include BN affine parameters too.
        # Exclusion from inner-loop adaptation is handled separately.
        root_params = OrderedDict(
            (name, param)
            for name, param in root_model.named_parameters()
            if name.startswith(("encoder.", "classifier."))
        )
        if not root_params:
            raise ValueError("Root model has no encoder/classifier parameters.")

        self.parameter_names = tuple(root_params.keys())

        # ParameterList avoids altering names containing dots.
        self.anchors = nn.ModuleList([
            nn.ParameterList([
                nn.Parameter(
                    param.detach().clone(),
                    requires_grad=param.requires_grad,
                )
                for param in root_params.values()
            ])
            for _ in range(n_anchors)
        ])

    @property
    def n_anchors(self):
        return len(self.anchors)

    def forward(self, indices, weights):
        """
        Build initial parameters for ONE task.

        indices: [top_k], selected anchor IDs
        weights: [top_k], corresponding normalized router weights

        Returns:
            OrderedDict with original encoder/classifier parameter names.
        """
        if indices.ndim != 1 or weights.ndim != 1:
            raise ValueError("Pass one task's indices and weights.")
        if indices.numel() == 0 or indices.numel() != weights.numel():
            raise ValueError("Indices and weights must have equal nonzero size.")
        if indices.dtype != torch.long:
            raise ValueError("indices must have dtype torch.long.")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Weights must be finite and nonnegative.")
        if not torch.allclose(
            weights.sum(), weights.new_tensor(1.0),
            atol=1e-6, rtol=1e-5,
        ):
            raise ValueError("Weights must sum to one.")

        anchor_ids = indices.detach().cpu().tolist()
        if len(set(anchor_ids)) != len(anchor_ids):
            raise ValueError("Anchor indices must be unique.")
        if any(k < 0 or k >= self.n_anchors for k in anchor_ids):
            raise ValueError("Anchor index out of range.")

        reference = self.anchors[anchor_ids[0]][0]
        if weights.device != reference.device:
            raise ValueError("Weights and anchor bank must be on the same device.")

        # Routing is fixed; gradients flow only into anchor parameters.
        fixed_weights = weights.detach()

        mixed = OrderedDict()
        for position, name in enumerate(self.parameter_names):
            value = None

            for j, anchor_id in enumerate(anchor_ids):
                param = self.anchors[anchor_id][position]
                term = fixed_weights[j].to(dtype=param.dtype) * param
                value = term if value is None else value + term

            mixed[name] = value

        return mixed