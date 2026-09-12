"""Soft anchor initialization with the existing MAML adaptation loop."""

import torch
import torch.nn as nn

from .maml import load as load_maml
from .soft_router import SoftRouter
from .anchor_bank import AnchorBank
from copy import deepcopy

class SoftAnchorModel(nn.Module):
    def __init__(
        self,
        root_ckpt,
        encoder_ckpt,
        geometry_path,
        top_k=2,
        tau=0.5,
    ):

        super().__init__()

        checkpoint = (
            root_ckpt
            if isinstance(root_ckpt, dict)
            else torch.load(
                root_ckpt,
                map_location="cpu",
                weights_only=False,
            )
        )

        # Load both encoder and classifier from the universal checkpoint.
        self.meta_model = load_maml(checkpoint, load_clf=True)

        # Current implementation uses the agreed ResNet-18/logistic setup.
        if checkpoint["encoder"] != "resnet18":
            raise ValueError("This first version requires resnet18.")
        if checkpoint["classifier"] != "logistic":
            raise ValueError("This first version requires logistic classifier.")
        if getattr(self.meta_model.classifier, "learn_temp", False):
            raise ValueError("Learnable classifier temperature is not supported.")

        for module in self.meta_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                if module.track_running_stats:
                    raise ValueError(
                        "This first version requires "
                        "BatchNorm track_running_stats=False."
                    )

        self.router = SoftRouter(
            encoder_ckpt=encoder_ckpt,
            geometry_path=geometry_path,
            top_k=top_k,
            tau=tau,
        )

        # IMPORTANT: create trainable copies BEFORE freezing the root.
        self.anchor_bank = AnchorBank(
            root_model=self.meta_model,
            n_anchors=self.router.n_anchors,
        )
        self.meta_model.requires_grad_(False)

        # Detached diagnostics for later W&B integration.
        self.last_routing = None

    @classmethod
    def from_checkpoint(cls, checkpoint):
        if checkpoint.get("checkpoint_version") != 1:
            raise ValueError("Unsupported soft-anchor checkpoint version.")

        state = checkpoint["model_state_dict"]
        settings = checkpoint["soft_anchor_args"]

        def extract(prefix):
            values = {
                name[len(prefix):]: tensor.detach().cpu()
                for name, tensor in state.items()
                if name.startswith(prefix)
            }
            if not values:
                raise ValueError("Missing checkpoint section: " + prefix)
            return values

        # Reconstruct the frozen root from the saved model state.
        root = {
            key: deepcopy(checkpoint[key])
            for key in (
                "encoder", "encoder_args",
                "classifier", "classifier_args",
            )
        }
        root["encoder_state_dict"] = extract(
            "meta_model.encoder."
        )
        root["classifier_state_dict"] = extract(
            "meta_model.classifier."
        )

        # The routing encoder may have different architecture arguments
        # from the meta-model, so use its own saved specification.
        router_checkpoint = deepcopy(
            checkpoint["router_encoder_spec"]
        )
        router_checkpoint["encoder_state_dict"] = extract(
            "router.encoder."
        )

        geometry = {
            "mean": state["router.pca_mean"].detach().cpu().numpy(),
            "components": (
                state["router.pca_components"].detach().cpu().numpy()
            ),
            "whitening_scale": (
                state["router.whitening_scale"].detach().cpu().numpy()
            ),
            "centers": state["router.centers"].detach().cpu().numpy(),
        }

        model = cls(
            root_ckpt=root,
            encoder_ckpt=router_checkpoint,
            geometry_path=geometry,
            top_k=settings["top_k"],
            tau=settings["tau"],
        )

        if model.n_anchors != settings["n_anchors"]:
            raise ValueError("Anchor count does not match checkpoint.")
        if list(model.anchor_bank.parameter_names) != list(
                checkpoint["anchor_parameter_names"]
        ):
            raise ValueError("Anchor parameter ordering does not match.")

        # Restore every saved tensor, including all trained anchors.
        model.load_state_dict(state, strict=True)
        return model
    @property
    def n_anchors(self):
        return self.anchor_bank.n_anchors

    def train(self, mode=True):
        super().train(mode)
        self.router.eval()
        return self

    def go_efficient(self, mode=True):
        self.meta_model.go_efficient(mode)
        return self

    def get_nan_grad_stats(self, reset=True):
        return self.meta_model.get_nan_grad_stats(reset=reset)

    def get_gradient_transport_gates(self, frozen=()):
        return {}

    def reset_classifier(self):
        raise ValueError(
            "SoftAnchorModel requires reset_classifier=False "
            "to preserve the learned anchor classifiers."
        )

    def forward(
        self,
        x_shot,
        x_query,
        y_shot,
        inner_args,
        meta_train,
        use_gradient_transport=False,
    ):
        """
        Same forward interface as the existing MAML model.

        Inputs:
            x_shot:  [episodes, support, C, H, W]
            x_query: [episodes, query, C, H, W]
            y_shot:  [episodes, support]

        Returns:
            logits: [episodes, query, n_way]
        """
        if use_gradient_transport:
            raise ValueError("Gradient transport must be disabled.")
        if not inner_args["first_order"]:
            raise ValueError("This experiment requires first_order=True.")
        if inner_args.get("reset_classifier", False):
            raise ValueError("reset_classifier must be False.")

        if x_shot.ndim != 5 or x_query.ndim != 5:
            raise ValueError("Support and query must have five dimensions.")
        if y_shot.ndim != 2:
            raise ValueError("Support labels must have two dimensions.")
        if x_shot.shape[:2] != y_shot.shape:
            raise ValueError("Support images and labels do not match.")
        if x_shot.shape[0] != x_query.shape[0]:
            raise ValueError("Support/query episode counts do not match.")
        if not (
            x_shot.device == x_query.device == y_shot.device
        ):
            raise ValueError("All inputs must be on the same device.")

        # Routing uses support only and never builds a gradient graph.
        routing = self.router(x_shot)
        self.last_routing = {
            key: routing[key].detach()
            for key in (
                "weights",
                "indices",
                "selected_weights",
                "distances",
            )
        }

        logits = []
        for episode in range(x_shot.shape[0]):
            # Build a separate initialization for each task.
            with torch.set_grad_enabled(meta_train):
                initial_params = self.anchor_bank(
                    routing["indices"][episode],
                    routing["selected_weights"][episode],
                )

            # During validation/test this method locally enables gradients
            # for support adaptation, without updating the anchor bank.
            task_logits = self.meta_model.forward_with_params(
                x_shot=x_shot[episode],
                x_query=x_query[episode],
                y_shot=y_shot[episode],
                initial_params=initial_params,
                inner_args=inner_args,
                meta_train=meta_train,
                episode=episode,
            )
            logits.append(task_logits)

        return torch.stack(logits, dim=0)