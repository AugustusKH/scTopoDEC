import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.mixture import GaussianMixture
from torch.distributions import Normal

logger = logging.getLogger(__name__)


class CPosterior:
    """
    Manages the post-training evaluations, embedding extractions, 
    and gene expression imputation mechanics independently of scvi-tools.
    """
    def __init__(self, model: torch.nn.Module, gene_dataset):
        self.model = model
        self.gene_dataset = gene_dataset
        self.indices = np.arange(gene_dataset.X.shape[0])
        self.nb_cells = len(self.indices)
        self.to_monitor = ["elbo"]

    @torch.no_grad()
    def get_latent(self, give_mean: Optional[bool] = True) -> Tuple[np.ndarray, np.ndarray]:
        """Output posterior z mean or sample, and cluster prediction label."""
        self.model.eval()
        device = next(self.model.parameters()).device
        latent = []
        labels_pred = []
        
        for tensors in self.gene_dataset:
            sample_batch, _, _, batch_index, _ = tensors
            sample_batch = sample_batch.to(device)
            if batch_index is not None:
                batch_index = batch_index.to(device)
                
            z, qc = self.model.sample_from_posterior_z(
                sample_batch, batch_index, give_mean=give_mean
            )
            latent.append(z.cpu())
            labels_pred.append(qc.argmax(dim=1).cpu())
            
        return (
            np.concatenate([z.numpy() for z in latent], axis=0),
            np.concatenate([l.numpy() for l in labels_pred], axis=0)
        )

    @torch.no_grad()
    def elbo(self) -> float:
        """Returns the Evidence Lower Bound associated to the object."""
        elbo_val = float(self.compute_elbo())
        logger.debug("ELBO : %.4f" % elbo_val)
        return elbo_val

    @torch.no_grad()
    def compute_elbo(self) -> float:
        """Computes the exact ELBO over the dataset partitions."""
        self.model.eval()
        device = next(self.model.parameters()).device
        elbo_accum = 0.0
        n_samples = len(self.indices)
        
        for tensors in self.gene_dataset:
            sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
            sample_batch = sample_batch.to(device)
            local_l_mean = local_l_mean.to(device)
            local_l_var = local_l_var.to(device)
            if batch_index is not None:
                batch_index = batch_index.to(device)
                
            reconst_loss, kl_divergence, kl_global = self.model(
                sample_batch, local_l_mean, local_l_var, batch_index=batch_index
            )
            elbo_accum += torch.sum(reconst_loss + kl_divergence).item()
            
            if isinstance(kl_global, torch.Tensor):
                elbo_accum += kl_global.item()
            else:
                elbo_accum += kl_global
                
        return float(elbo_accum / n_samples)

    @torch.no_grad()
    def imputation(
            self,
            n_samples: Optional[int] = 1,
            transform_batch: Optional[Union[int, List[int]]] = None,
    ) -> np.ndarray:
        """Imputes px_rate values mapped over cellular profiles."""
        self.model.eval()
        device = next(self.model.parameters()).device
        if (transform_batch is None) or (isinstance(transform_batch, int)):
            transform_batch = [transform_batch]
            
        imputed_arr = []
        for batch in transform_batch:
            imputed_list_batch = []
            for tensors in self.gene_dataset:
                sample_batch, _, _, batch_index, _ = tensors
                sample_batch = sample_batch.to(device)
                if batch_index is not None:
                    batch_index = batch_index.to(device)
                    
                px_rate = self.model.get_sample_rate(
                    sample_batch,
                    batch_index=batch_index,
                    n_samples=n_samples,
                    transform_batch=batch,
                )
                imputed_list_batch.append(px_rate.cpu().numpy())
            imputed_arr.append(np.concatenate(imputed_list_batch, axis=0))
            
        imputed_arr = np.array(imputed_arr)
        return imputed_arr.mean(0).squeeze()


class CTrainer:
    """
    A unified execution environment managing training loops, learning optimization,
    and Gaussian Mixture warm-starts on targeted text datasets.
    """
    def __init__(self, model, gene_dataset, train_size=1.0, use_cuda=True,
                 n_epochs_kl_warmup=200, n_epochs_pre_train=200, tol=0.001,
                 tol_start=10, extra_steps=1, **kwargs):
        self.model = model
        self.gene_dataset = gene_dataset
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Partition monitors
        self.train_set = CPosterior(self.model, self.gene_dataset)
        self.test_set = self.train_set
        self._posteriors = {"train_set": self.train_set}
        
        self.n_samples = len(self.train_set.indices)
        self.n_epochs_pre_train = 0 if n_epochs_pre_train is None else n_epochs_pre_train
        self.n_epochs_kl_warmup = 0 if n_epochs_kl_warmup is None else n_epochs_kl_warmup
        self.extra_steps = extra_steps
        self.clustering = None
        self.tol = tol
        self.tol_start = tol_start
        self.epoch = 0
        self.kl_weight = 1.0
        self.normalize_loss = True
        self.frequency = 10
        self.history = {"elbo_train_set": []}

    def train(self, n_epochs: int = 400, lr: float = 0.001, **kwargs):
        """Executes the standard structural training loops over the parameters."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
        for epoch in range(n_epochs):
            self.epoch = epoch
            
            # Step A: Check pretrain/clustering transitions
            self.on_epoch_begin()
            
            # Step B: Anneal the KL weights linearly
            if self.n_epochs_kl_warmup > 0:
                self.kl_weight = min(1.0, epoch / self.n_epochs_kl_warmup)
            else:
                self.kl_weight = 1.0
                
            self.model.train()
            epoch_loss = 0.0
            
            # Step C: Mini-batch execution pass
            for tensors in self.gene_dataset:
                sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
                
                sample_batch = sample_batch.to(self.device)
                local_l_mean = local_l_mean.to(self.device)
                local_l_var = local_l_var.to(self.device)
                if batch_index is not None:
                    batch_index = batch_index.to(self.device)
                    
                self.optimizer.zero_grad()
                
                reconst_loss, kl_local, kl_global = self.model(
                    sample_batch, local_l_mean, local_l_var, batch_index
                )
                
                loss = torch.mean(reconst_loss + self.kl_weight * kl_local) + torch.sum(kl_global)
                if self.normalize_loss:
                    loss = loss / self.n_samples
                    
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
            # Step D: Run validation monitoring hooks
            continue_training = self.on_epoch_end()
            if not continue_training:
                print(f"Early optimization termination triggered at epoch {epoch+1}")
                break

    @torch.no_grad()
    def compute_running_parameters(self, z=None, qc=None, tol=0.001, tol_start=None) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Dynamically computes spatial cluster updates and tracks changes."""
        continue_training = True
        tol_start = self.tol_start if tol_start is None else tol_start
        
        if z is None:
            z_ = []
            qc_ = []
            self.model.eval()
            for tensors in self.gene_dataset:
                sample_batch, _, _, batch_index, _ = tensors
                sample_batch = sample_batch.to(self.device)
                if batch_index is not None:
                    batch_index = batch_index.to(self.device)

                if self.model.log_variational:
                    sample_batch = torch.log(1 + sample_batch)

                _, _, z_val = self.model.z_encoder(sample_batch, batch_index)
                qc_val = self.model.classifier(z_val)

                z_.append(z_val)
                qc_.append(qc_val)

            z = torch.cat(z_, dim=0)
            qc = torch.cat(qc_, dim=0)
            
        if tol >= 0 and tol_start <= self.epoch + 2:
            clustering = qc.argmax(dim=-1)
            if self.clustering is not None:
                delta_clustering = sum(clustering != self.clustering).item() / len(clustering)
                if delta_clustering < tol:
                    logger.info(f"Tolerance achieved : ratio of clustering labels changed under {tol}")
                    continue_training = False
                else:
                    self.clustering = clustering
            else:
                self.clustering = clustering

        if continue_training:
            qc_sum = qc.sum(dim=0, keepdim=True)
            self.model.mu = torch.nn.Parameter(qc.t().mm(z) / qc_sum.t())
            self.model.pi = torch.nn.Parameter(qc_sum / qc.shape[0])

        return z, qc, continue_training

    @torch.no_grad()
    def more_steps_for_gmm(self, z=None, qc=None, max_steps=1, tol=0.001) -> bool:
        continue_training = True
        for i in range(max_steps):
            z, qc, continue_training = self.compute_running_parameters(z, qc, tol)
            if not continue_training:
                break
        return continue_training

    @torch.no_grad()
    def on_epoch_begin(self):
        """Warm start step converting continuous profiles into structured mixtures."""
        if self.n_epochs_pre_train == self.epoch and self.epoch > 0:
            print("Pretraining epoch limit reached. Initializing global GMM partitions...")
            self.model.eval()
            z_ = []
            for tensors in self.gene_dataset:
                sample_batch, _, _, batch_index, _ = tensors
                sample_batch = sample_batch.to(self.device)
                if batch_index is not None:
                    batch_index = batch_index.to(self.device)

                if self.model.log_variational:
                    sample_batch = torch.log(1 + sample_batch)

                _, _, z_val = self.model.z_encoder(sample_batch, batch_index)
                z_.append(z_val.cpu())
                
            z = torch.cat(z_, dim=0).detach().numpy()

            if not self.model.n_labels:
                data = anndata.AnnData(X=z)
                sc.pp.neighbors(data)
                sc.tl.louvain(data, resolution=self.model.resolution)
                self.model.n_labels = len(np.unique(data.obs['louvain']))

            gmm = GaussianMixture(n_components=self.model.n_labels).fit(z)
            mean_ = torch.from_numpy(gmm.means_).to(torch.float32).to(self.device)
            weights_ = torch.from_numpy(gmm.weights_).to(torch.float32).to(self.device)
            
            self.model.mu = torch.nn.Parameter(mean_)
            self.model.pi = torch.nn.Parameter(weights_)
            self.model.pretrain = False

    @torch.no_grad()
    def on_epoch_end(self) -> bool:
        continue_training = True
        if self.n_epochs_pre_train <= self.epoch:
            continue_training = self.more_steps_for_gmm(max_steps=self.extra_steps, tol=self.tol)
            
        if not continue_training:
            return continue_training
            
        if self.frequency and (self.epoch + 1) % self.frequency == 0:
            self.compute_metrics()
            
        return True

    @torch.no_grad()
    def compute_metrics(self):
        """Tracks optimization variables across training sets."""
        self.model.eval()
        epoch_idx = self.epoch + 1
        logger.debug(f"\nEPOCH [{epoch_idx}]: Checking tracking targets...")
        
        current_elbo = self.train_set.elbo()
        self.history["elbo_train_set"].append(current_elbo)
        print(f"Epoch [{epoch_idx}] -> Current ELBO Log Metrics: {current_elbo:.4f}")