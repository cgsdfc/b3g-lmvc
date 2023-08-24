"""
Batched Bag of Bipartite Graphs for Large-scale Multi-view Clustering
(B3LMVC)
"""
import copy
import logging
from pathlib import Path as P
from typing import List
import random

import torch
from joblib import Parallel, delayed
import time
import math

from src.utils.torch_utils import *
from src.utils.metrics import Evaluate_Graph, MaxMetrics, KMeans_Evaluate
from sklearn.cluster import KMeans
from src.vis.visualize import *
import time

from src.utils.io_utils import train_begin, train_end, save_var
from src.utils.torch_utils import *
from src.vis.visualize import *

from .utils import compute_num_anchors


def make_anchor_mask_numpy(num_samples: int, num_anchors: int):
    anchor_idx = random.sample(range(num_samples), k=num_anchors)
    anchor_mask = np.zeros(num_samples, dtype=bool)
    anchor_mask[anchor_idx] = 1
    return anchor_mask


class MinibatchFastModel(nn.Module):
    def __init__(
        self,
        num_samples: int,
        num_anchors: int,
        num_views: int,
        num_bags: int,
        temperature: float,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_anchors = num_anchors
        self.num_views = num_views
        self.num_bags = num_bags
        self.temperature = temperature
        params = nn.Parameter(
            torch.zeros(num_samples, num_anchors), requires_grad=False
        )
        self.Z_params = params

    def fit(
        self,
        X_list: List[Tensor],
        data: MultiviewDataset,
        device: int,
        epochs: int,
        valid_freq=10,
        save_history=False,
        save_history_vars: bool = False,
        save_vars: bool = False,
        save_history_metrics: bool = False,
        verbose=True,
    ):
        history = []
        outputs = {}
        mm = MaxMetrics()
        T = 0
        for epoch in range(epochs):
            epoch_begin = time.time()
            self.train()
            loss = self.compute_loss_bag(X_list, device, epoch)
            epoch_time = time.time() - epoch_begin
            T += epoch_time

            if (1 + epoch) % valid_freq == 0 and verbose:
                Z_common = self.Z_common(epoch)
                if save_history_metrics:
                    metrics, ff = Evaluate_Graph(
                        data=data,
                        Z=Z_common,
                        type="fastSVD",
                        return_spectral_embedding=True,
                    )
                    print(
                        f"epoch {epoch:04d} loss {float(loss):.6f} {metrics} T: {epoch_time}"
                    )
                else:
                    metrics = {}
                    ff = None

                if mm.update(**metrics)["ACC"] and save_vars and save_history_metrics:
                    outputs.update(
                        Z_common=convert_numpy(Z_common), H_common=convert_numpy(ff)
                    )
                if save_history:
                    metrics["loss"] = float(loss)
                    if save_history_vars:
                        metrics["Z_common"] = convert_numpy(Z_common)
                    history.append(metrics)

        mm.update(T=T)
        outputs["history"] = history
        outputs["metrics"] = mm.report(current=False)
        return outputs

    def compute_loss_bag(self, X_list, device, epoch: int):
        idx_anchors = self.create_masks(device)
        Z_new = torch.zeros(self.num_samples, self.num_anchors).to(device)
        for v in range(self.num_views):
            X = convert_tensor(X_list[v], dev=device)
            A = torch.stack([X[idx].T for idx in idx_anchors], 0).to(device)
            # U = torch.stack([X[~idx] for idx in idx_anchors], 0).to(device)
            # W = torch.stack([torch.eye(self.num_samples)[idx] for idx in idx_anchors], 0).to(device)
            X = torch.tile(X, [self.num_bags, 1, 1])
            Z = F.softmax(X @ A / self.temperature, 2)
            Z = torch.mean(Z, 0)
            Z_new += Z
        Z_new /= self.num_views
        loss = (
            ((self.Z_params + Z_new) / (epoch + 2) - self.Z_params / (epoch + 1))
            .pow(2)
            .sum()
        )
        loss /= self.num_samples
        self.Z_params += Z_new
        return loss

    def create_masks(self, device):
        anchor_mask_list = Parallel(n_jobs=-1, backend="threading")(
            delayed(make_anchor_mask_numpy)(self.num_samples, self.num_anchors)
            for _ in range(self.num_bags)
        )
        return anchor_mask_list

    @torch.no_grad()
    def Z_common(self, epoch: int) -> Tensor:
        self.eval()
        return self.Z_params / (epoch + 1)


def train_main(
    datapath=P("./data/ORL-40.mat"),
    views=None,
    method: str = "b3lmvc",
    # minibatch
    num_bags=10,
    epochs=100,
    temperature: float = 0.1,
    valid_freq=10,
    device=get_device(),
    savedir: P = P("output/debug"),
    save_vars: bool = False,
    save_history: bool = False,
    save_history_vars: bool = False,
    save_history_metrics: bool = True,
):
    config = dict(
        datapath=datapath,
        views=views,
        method=method,
        temperature=temperature,
    )
    train_begin(savedir, config, f"Begin train {method}")

    data = MultiviewDataset(
        datapath=datapath,
        view_ids=views,
        normalize="l2",
    )
    print(data.describe())

    num_anchors = compute_num_anchors(data.sampleNum, data.clusterNum)

    model = MinibatchFastModel(
        num_samples=data.sampleNum,
        num_anchors=num_anchors,
        num_views=data.viewNum,
        num_bags=num_bags,
        temperature=temperature,
    ).to(device)
    outputs = model.fit(
        X_list=data.X,
        device=device,
        epochs=epochs,
        data=data,
        valid_freq=valid_freq,
        verbose=True,
        save_history=save_history,
        save_vars=save_vars,
        save_history_vars=save_history_vars,
        save_history_metrics=save_history_metrics,
    )

    train_end(savedir, outputs["metrics"])
    if save_vars:
        save_var(savedir, outputs["H_common"], "H_common")
        save_var(savedir, outputs["Z_common"], "Z_common")

    if save_history:
        save_var(savedir, outputs["history"], "history")
