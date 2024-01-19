from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from scipy.spatial.transform import Rotation as R
from typing import List, Callable, Any, Dict
import os


@register_loss("unimolv2")
class Unimolv2Loss(UnicoreLoss):
    """
    Implementation for the loss used in masked graph model (MGM) training.
    """

    def __init__(self, task):
        super().__init__(task)
        self.args = task.args

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        with torch.no_grad():
            sample_size = sample["batched_data"]["atom_feat"].shape[0]
            natoms = sample["batched_data"]["atom_feat"].shape[1]

        # add gaussian noise
        (
            graph_output,
            pos_pred,
            dist_pred,
            plddt_logits,
        ) = model(**sample)
        targets = sample["batched_data"]["target"].float().view(-1)
        per_data_loss = None
        if graph_output is not None:
            per_data_loss = torch.nn.L1Loss(reduction="none")(
                graph_output.float(), targets
            )
            loss = per_data_loss.sum()
        else:
            loss = torch.tensor(0.0, device=targets.device)

        atom_mask = sample["batched_data"]["atom_mask"].float()
        pos_mask = atom_mask.unsqueeze(-1)
        pos_target = sample["batched_data"]["pos_target"].float() * pos_mask

        def get_pos_loss(pos_pred):
            pos_pred = pos_pred.float() * pos_mask
            center_loss = pos_pred.mean(dim=-2).square().sum()
            pos_loss = torch.nn.L1Loss(reduction="none")(
                pos_pred,
                pos_target,
            ).sum(dim=(-1, -2))
            pos_cnt = pos_mask.squeeze(-1).sum(dim=-1) + 1e-10
            pos_loss = (pos_loss / pos_cnt).sum()
            return pos_loss, center_loss

        (pos_loss, center_loss) = get_pos_loss(pos_pred)

        pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2).float()
        dist_target = (pos_target.unsqueeze(-2) - pos_target.unsqueeze(-3)).norm(dim=-1)
        dist_target = dist_target * pair_mask
        dist_cnt = pair_mask.sum(dim=(-1, -2)) + 1e-10

        def get_dist_loss(dist_pred, return_sum=True):
            dist_pred = dist_pred.float() * pair_mask
            dist_loss = torch.nn.L1Loss(reduction="none")(
                dist_pred,
                dist_target,
            ).sum(dim=(-1, -2))
            if return_sum:
                return (dist_loss / dist_cnt).sum()
            else:
                return dist_loss / dist_cnt

        dist_loss = get_dist_loss(dist_pred)

        plddt_logits = plddt_logits.float()
        cutoff = 15.0
        num_bins = 50
        eps = 1e-10
        lddt = self.compute_lddt(
            dist_pred.float(),
            dist_target,
            pair_mask,
            cutoff=cutoff,
            eps=eps,
        ).detach()

        bin_index = torch.floor(lddt * num_bins).long()
        bin_index = torch.clamp(bin_index, max=(num_bins - 1))
        lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=num_bins)
        errors = self.softmax_cross_entropy(plddt_logits, lddt_ca_one_hot)
        plddt_loss = self.masked_mean(atom_mask, errors, dim=-1, eps=eps).sum()
        ca_lddt = self.masked_mean(atom_mask, lddt, dim=-1, eps=eps)
        plddt = self.masked_mean(
            atom_mask, self.predicted_lddt(plddt_logits), dim=-1, eps=eps
        )

        total_loss = (
            loss
            + dist_loss
            + self.args.pos_loss_weight * (pos_loss + center_loss)
            + self.args.plddt_loss_weight * plddt_loss
        )
        logging_output = {
            "loss": loss.data,
            "dist_loss": dist_loss.data,
            "pos_loss": pos_loss.data,
            "center_loss": center_loss.data,
            "total_loss": total_loss.data,
            "plddt_loss": plddt_loss.data,
            "ca_lddt_metric": ca_lddt.sum().data,
            "plddt_metric": plddt.sum().data,
            "sample_size": sample_size,
            "nsentences": sample_size,
            "ntokens": natoms,
            "bsz": sample_size,
            "n_atoms": natoms * sample_size,
        }
        if not torch.is_grad_enabled():
            logging_output["id"] = sample["batched_data"]["id"].cpu().numpy()
            if per_data_loss is None:
                per_data_loss = 1.0 - ca_lddt
            logging_output["per_data"] = per_data_loss.detach().cpu().numpy()
            logging_output["plddt"] = plddt.detach().cpu().numpy()
            logging_output["ca_lddt"] = ca_lddt.detach().cpu().numpy()
        logging_output["is_pretrain"] = 1.0 if graph_output is None else 0.0
        logging_output["total_loss"] = total_loss.data
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""
        is_pretrain = sum(log.get("is_pretrain", 0) for log in logging_outputs) > 0
        if split != "train":
            prefix = "" if not is_pretrain else "pretrain_"
            id = np.concatenate([log["id"] for log in logging_outputs])
            per_data = np.concatenate([log["per_data"] for log in logging_outputs])
            plddt = np.concatenate([log["plddt"] for log in logging_outputs])
            ca_lddt = np.concatenate([log["ca_lddt"] for log in logging_outputs])
            df = pd.DataFrame(
                {
                    "id": id,
                    "loss": per_data,
                    "plddt": plddt,
                    "ca_lddt": ca_lddt,
                }
            )
            df_grouped = df.groupby(["id"])
            df_min = df_grouped.agg("min")
            df_mean = df_grouped.agg("mean")
            df_median = df_grouped.agg("median")
            df_plddt = (
                df.sort_values(by=["id", "plddt"], ascending=[True, False])
                .groupby("id", as_index=False)
                .head(1)
            )
            df_ca_lddt = (
                df.sort_values(by=["id", "ca_lddt"], ascending=[True, False])
                .groupby("id", as_index=False)
                .head(1)
            )
            assert len(df_min["loss"]) == len(df_plddt["loss"])
            metrics.log_scalar(
                prefix + "loss_by_plddt", df_plddt["loss"].mean(), 1, round=6
            )
            metrics.log_scalar(
                prefix + "loss_by_ca_lddt", df_ca_lddt["loss"].mean(), 1, round=6
            )
            metrics.log_scalar(
                prefix + "loss_by_min", df_min["loss"].mean(), 1, round=6
            )
            metrics.log_scalar(prefix + "loss_cnt", len(df_min["loss"]), 1, round=6)
            metrics.log_scalar(
                prefix + "loss_by_mean", df_mean["loss"].mean(), 1, round=6
            )
            metrics.log_scalar(
                prefix + "loss_by_median", df_median["loss"].mean(), 1, round=6
            )

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        n_atoms = sum(log.get("n_atoms", 0) for log in logging_outputs)
        for key in logging_outputs[0].keys():
            if "loss" in key or "metric" in key:
                total_loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key, total_loss_sum / sample_size, sample_size, round=6
                )
        metrics.log_scalar("n_atoms", n_atoms / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train

    def compute_lddt(
        self,
        dmat_pred,
        dmat_true,
        pair_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        n = pair_mask.shape[-1]
        dists_to_score = (
            (dmat_true < cutoff)
            * pair_mask
            * (1.0 - torch.eye(n, device=pair_mask.device))
        )

        dist_l1 = torch.abs(dmat_true - dmat_pred)

        score = (
            (dist_l1 < 0.05).type(dist_l1.dtype)
            + (dist_l1 < 0.1).type(dist_l1.dtype)
            + (dist_l1 < 0.2).type(dist_l1.dtype)
            + (dist_l1 < 0.4).type(dist_l1.dtype)
        )
        score = score * 0.25

        norm = 1.0 / (eps + torch.sum(dists_to_score, dim=-1))
        score = norm * (eps + torch.sum(dists_to_score * score, dim=-1))
        return score

    def masked_mean(self, mask, value, dim, eps=1e-10, keepdim=False):
        mask = mask.expand(*value.shape)
        return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (
            eps + torch.sum(mask, dim=dim, keepdim=keepdim)
        )

    def softmax_cross_entropy(self, logits, labels):
        loss = -1 * torch.sum(
            labels * torch.nn.functional.log_softmax(logits.float(), dim=-1),
            dim=-1,
        )
        return loss

    def predicted_lddt(self, plddt_logits: torch.Tensor) -> torch.Tensor:
        """Computes per-residue pLDDT from logits.
        Args:
            logits: [num_res, num_bins] output from the PredictedLDDTHead.
        Returns:
            plddt: [num_res] per-residue pLDDT.
        """
        num_bins = plddt_logits.shape[-1]
        bin_probs = torch.nn.functional.softmax(plddt_logits.float(), dim=-1)
        bin_width = 1.0 / num_bins
        bounds = torch.arange(
            start=0.5 * bin_width, end=1.0, step=bin_width, device=plddt_logits.device
        )
        plddt = torch.sum(
            bin_probs * bounds.view(*((1,) * len(bin_probs.shape[:-1])), *bounds.shape),
            dim=-1,
        )
        return plddt
