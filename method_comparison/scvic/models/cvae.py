import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl

from scvic.models.modules import DecoderList, EncoderList
from scvic.utils import log_nb_positive, log_zinb_positive

torch.backends.cudnn.benchmark = True
logger = logging.getLogger(__name__)



# ==============================================================================
# Native Replacements for Legacy scvi.models.utils Utilities
# ==============================================================================
def one_hot(index: torch.Tensor, n_cats: int) -> torch.Tensor:
    """Converts a class index tensor into a standard one-hot tensor."""
    onehot = torch.zeros(index.size(0), n_cats, device=index.device)
    onehot.scatter_(1, index.view(-1, 1).long(), 1)
    return onehot

def broadcast_labels(y: Optional[torch.Tensor], z: torch.Tensor, n_broadcast: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcasts the latent representation 'z' across all possible cluster labels
    so the GMM classifier can evaluate cluster densities concurrently.
    """
    n_samples = z.size(0)
    
    # Create an artificial label matrix containing every single cluster ID repeated for each sample
    labels = torch.arange(n_broadcast, device=z.device).view(-1, 1).repeat(1, n_samples).view(-1, 1)
    one_hot_labels = one_hot(labels, n_broadcast)
    
    # Expand the latent space tensor to match the shape of the full label matrix
    z_broadcast = z.repeat(n_broadcast, 1)
    
    return one_hot_labels, z_broadcast
    


class CVAE(nn.Module):

    def __init__(
            self,
            n_input: int,
            n_labels: Optional[int] = None,
            resolution: Union[float, int, None] = 1.0,
            n_hidden: Union[int, List[int]] = 128,
            n_layers: Optional[int] = 1,
            n_batch: int = 0,
            n_latent: int = 10,
            dropout_rate: float = 0.1,
            dispersion: str = "gene",
            log_variational: bool = True,
            reconstruction_loss: str = "zinb"
    ):
        super().__init__()
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
            
        if len(n_hidden) > 1 and n_layers > 1:
            logger.warning(
                "Hidden node number list has been given, hidden layer number is ignored. "
                "Now hidden node number list is " + str(n_hidden)
            )
        else:
            n_hidden = n_hidden * n_layers
            
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.resolution = resolution

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-cell":
            pass
        else:
            # FIXED: Corrected syntax bug where format was enclosed inside the string literals
            raise ValueError(
                f"dispersion must be one of ['gene', 'gene-batch', 'gene-label', 'gene-cell'], "
                f"but input was {self.dispersion}"
            )

        self.classifier = self.classify

        # z encoder: maps n_input features to the n_latent bottleneck space
        self.z_encoder = EncoderList(
            n_input,
            n_latent,
            n_hidden=n_hidden,
            n_cat_list=[n_batch],
            dropout_rate=dropout_rate
        )
        # l encoder: maps n_input features to a 1D library size factor 
        self.l_encoder = EncoderList(
            n_input, 1, n_hidden=n_hidden, n_cat_list=[n_batch], dropout_rate=dropout_rate,
        )
        # decoder: maps latent vectors back to the data count space
        self.decoder = DecoderList(
            n_latent,
            n_input,
            n_hidden=n_hidden[::-1],
            n_cat_list=[n_batch],
        )

        self.pretrain = True

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Calculates variational cluster responsibilities q(c|z) using Gaussian components.
        """
        cs, zs = broadcast_labels(None, z, n_broadcast=self.n_labels)
        
        # FIXED: Enforced strict device synchronization for dynamically assigned GMM parameters
        mu_device = self.mu.to(z.device)
        pi_device = self.pi.to(z.device)
        
        pz_mu_cs = torch.mm(cs, mu_device)
        pz_var_cs = torch.ones_like(pz_mu_cs)
        
        log_pz_c = Normal(pz_mu_cs, torch.sqrt(pz_var_cs)).log_prob(zs).sum(dim=-1).view(
            self.n_labels, -1
        ).t() + torch.log(pi_device + 1e-8)
        
        qc_z = F.softmax(log_pz_c, dim=-1)
        return qc_z

    def sample_from_posterior_z(
            self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, give_mean: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.log_variational:
            x = torch.log(1 + x)
        qz_m, qz_v, z = self.z_encoder(x, batch_index)
        if give_mean:
            z = qz_m
        qc = self.classifier(z)
        return z, qc

    def get_reconstruction_loss(
            self, x: torch.Tensor, px_rate: torch.Tensor, px_r: torch.Tensor, px_dropout: torch.Tensor, **kwargs
        ) -> torch.Tensor:
        if self.reconstruction_loss == "zinb":
            reconst_loss = -log_zinb_positive(x, px_rate, px_r, px_dropout).sum(dim=-1)
        elif self.reconstruction_loss == "nb":
            reconst_loss = -log_nb_positive(x, px_rate, px_r).sum(dim=-1)
        return reconst_loss

    def get_sample_rate(
            self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, n_samples: int = 1, transform_batch: Optional[int] = None
    ) -> torch.Tensor:
        return self.inference(
            x,
            batch_index=batch_index,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_rate"]

    def get_sample_scale_from_z(self, z: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        px = self.decoder.px_decoder(z, batch_index)
        px_scale = self.decoder.px_scale_decoder(px)
        return px_scale

    def get_sample_scale(self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, n_samples: int = 1, transform_batch: Optional[int] = None
                         ) -> torch.Tensor:
        return self.inference(
            x,
            batch_index=batch_index,
            n_samples=n_samples,
            transform_batch=transform_batch,
        )["px_scale"]

    def inference(
            self, x: torch.Tensor, batch_index: Optional[torch.Tensor] = None, n_samples: int = 1, transform_batch: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        x_ = x
        if self.log_variational:
            x_ = torch.log(1 + x_)

        qz_m, qz_v, z = self.z_encoder(x_, batch_index)
        ql_m, ql_v, library = self.l_encoder(x_, batch_index)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            library = Normal(ql_m, ql_v.sqrt()).sample()

        if transform_batch is not None:
            dec_batch_index = transform_batch * torch.ones_like(batch_index)
        else:
            dec_batch_index = batch_index

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion, z, library, dec_batch_index
        )
        
        if self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(dec_batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r) + 1e-4

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    def forward(
            self, x: torch.Tensor, local_l_mean: torch.Tensor, local_l_var: torch.Tensor, batch_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        outputs = self.inference(x, batch_index)
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        z = outputs["z"]
        ql_m = outputs["ql_m"]
        ql_v = outputs["ql_v"]
        px_rate = outputs["px_rate"]
        px_r = outputs["px_r"]
        px_dropout = outputs["px_dropout"]

        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_l_mean, torch.sqrt(local_l_var)),
        ).sum(dim=-1)

        if self.pretrain:
            mean = torch.zeros_like(qz_m)
            scale = torch.ones_like(qz_v)
            kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
            kl_divergence = kl_divergence_z
            reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout) + kl_divergence_l
        else:
            qc = self.classifier(z)
            
            # FIXED: Dynamically map the GMM parameters to the data execution device
            pi_device = self.pi.to(z.device)
            mu_device = self.mu.to(z.device)
            
            c_prior = pi_device.expand(qc.size())
            kl_divergence_c = kl(Categorical(qc), Categorical(c_prior))
            
            cs, zs = broadcast_labels(None, z, n_broadcast=self.n_labels)
            pz_mu_cs = torch.mm(cs, mu_device)
            pz_var_cs = torch.ones_like(pz_mu_cs)

            loss_z_weight = Normal(qz_m, torch.sqrt(qz_v)).log_prob(z).sum(dim=-1)

            loss_z_unweight = -(Normal(pz_mu_cs, torch.sqrt(pz_var_cs)).log_prob(zs).sum(
                dim=-1).view(self.n_labels, -1).t() * qc
            ).sum(dim=-1)
            
            kl_divergence = kl_divergence_l + kl_divergence_c
            reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout) + loss_z_unweight + loss_z_weight

        return reconst_loss, kl_divergence, 0.0