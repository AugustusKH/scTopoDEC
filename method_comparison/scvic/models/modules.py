import collections
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch import nn as nn
from torch.distributions import Normal
from scvi.models.utils import one_hot


def reparameterize_gaussian(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """
    Applies the VAE reparameterization trick to sample from a Gaussian distribution.
    """
    return Normal(mu, var.sqrt()).rsample()


class FCLayersList(nn.Module):
    r"""A helper class to build fully-connected layers for a neural network.

    :param n_in: The dimensionality of the input
    :param n_out: The dimensionality of the output
    :param n_hidden: The number of nodes per hidden layer
    :param n_cat_list: A list containing the number of categories per categorical covariate.
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    :param use_batch_norm: Whether to have `BatchNorm` layers or not
    :param use_relu: Whether to have `ReLU` layers or not
    :param bias: Whether to learn bias in linear layers or not
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_hidden: List[int],
        n_cat_list: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_relu: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        layers_dim = [n_in] + n_hidden + [n_out]

        if n_cat_list is not None:
            self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
        else:
            self.n_cat_list = []

        # Dynamically build layers to strictly avoid passing None into nn.Sequential
        dict_layers = collections.OrderedDict()
        for i, (layer_in, layer_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:])):
            sub_layers = []
            
            # Linear step expects concatenated category representations
            sub_layers.append(nn.Linear(layer_in + sum(self.n_cat_list), layer_out, bias=bias))
            
            if use_batch_norm:
                sub_layers.append(nn.BatchNorm1d(layer_out, momentum=0.01, eps=0.001))
            if use_relu:
                sub_layers.append(nn.ReLU())
            if dropout_rate > 0:
                sub_layers.append(nn.Dropout(p=dropout_rate))
                
            dict_layers["Layer {}".format(i)] = nn.Sequential(*sub_layers)

        self.fc_layers = nn.Sequential(dict_layers)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor, instance_id: int = 0) -> torch.Tensor:
        one_hot_cat_list = []
        assert len(self.n_cat_list) <= len(cat_list), "nb. categorical args provided doesn't match init. params."
        
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            assert not (n_cat and cat is None), "cat not provided while n_cat != 0 in init. params."
            if n_cat > 1:
                if cat.size(1) != n_cat:
                    one_hot_cat = one_hot(cat, n_cat)
                else:
                    one_hot_cat = cat
                one_hot_cat_list += [one_hot_cat]

        for layers_block in self.fc_layers:
            for layer in layers_block:
                if isinstance(layer, nn.BatchNorm1d):
                    if x.dim() == 3:
                        x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                    else:
                        x = layer(x)
                else:
                    if isinstance(layer, nn.Linear):
                        if x.dim() == 3:
                            one_hot_cat_list_layer = [
                                o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                for o in one_hot_cat_list
                            ]
                        else:
                            one_hot_cat_list_layer = one_hot_cat_list
                        x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                    x = layer(x)
        return x


# Encoder
class EncoderList(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: List[int],
        n_cat_list: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.encoder = FCLayersList(
            n_in=n_input,
            n_out=n_hidden[-1],
            n_hidden=n_hidden[:-1],
            n_cat_list=n_cat_list,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden[-1], n_output)
        self.var_encoder = nn.Linear(n_hidden[-1], n_output)

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


# Decoder
class DecoderList(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: List[int],
        n_cat_list: Optional[Iterable[int]] = None
    ):
        super().__init__()
        self.px_decoder = FCLayersList(
            n_in=n_input,
            n_out=n_hidden[-1],
            n_hidden=n_hidden[:-1],
            n_cat_list=n_cat_list,
            dropout_rate=0,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden[-1], n_output), nn.Softmax(dim=-1)
        )

        # dispersion
        self.px_r_decoder = nn.Linear(n_hidden[-1], n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden[-1], n_output)

    def forward(
        self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout