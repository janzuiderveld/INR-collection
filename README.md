# INR-collection
A collection of (conditional \ modulatable) implicit neural representation (INR) implementations and building blocks in PyTorch. 

[![PyPI version](https://badge.fury.io/py/INR-collection.svg)](https://badge.fury.io/py/INR-collection)

This package is aimed to help in quick prototyping for applying INRs to new domains.

Currently, the following conditioning methods are supported:
- Feature wise linear modulation (FiLM)
- Concatenation
- Post activation modulation (experimental)

Additionaly, several nonlinearities, weight initalization methods and progressive activation scaling in sinusoidal INRs are supported. Allowing easier "interpolation" between several prominent INR approaches e.g.
- Pi-GAN - <a href="https://openaccess.thecvf.com/content/CVPR2021/papers/Chan_Pi-GAN_Periodic_Implicit_Generative_Adversarial_Networks_for_3D-Aware_Image_Synthesis_CVPR_2021_paper.pdf"> Periodic implicit generative adversarial networks for 3d-aware image synthesis </a> 
- IM-NET - <a href="http://summit.sfu.ca/system/files/iritems1/19324/etd20312.pdf"> Learning implicit fields for generative shape modeling </a>
- DeepSDF - <a href="https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_DeepSDF_Learning_Continuous_Signed_Distance_Functions_for_Shape_Representation_CVPR_2019_paper.pdf"> Learning Continuous Signed Distance Functions
for Shape Representation </a>
- SIREN - <a href="https://arxiv.org/abs/2006.09661">Implicit Neural Representations with Periodic Activation Function</a>
- MFN - <a href="https://openreview.net/pdf?id=OmtmcPkkhT"> Multiplicative Filter Networks </a>

# Install 

```bash
$ pip install INR-collection
```

# Usage
We support directly callable implementations of Pi-GAN, IM-NET and SIREN.

```python
"""
Applying (a slightly simplified version of) Pi-GAN to images 
"""
import torch
from INR_collection import piGAN

in_features = 2 # two-dimensional coordinates
out_features = 3 # RGB

INR = piGAN(in_features, 
            out_features, 
            num_INR_layers=8,       # set INR depth
            num_hidden_INR=256,     # set INR width
            num_hidden_mapping=256, # set latent mapping network width 
            num_mapping_layers=3,   # set latent mapping network depth
            z_size=256,             # set latent embedding size
            first_omega_0=600,      # set activation scaling - first layer
            hidden_omega_0=30)      # - hidden layers


coord = torch.randn(1, 2)
INR(coord) # (1, 3) <- rgb value
```

For more customization; The main building block for these architectures can be imported as ImplicitMLPLayer, which has the following variables:

```python
class ImplicitMLPLayer(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                bias=True,
                omega_0=1, 
                w_norm=False, 
                activation="relu",                                # relu, sine, sigmoid, tanh, none
                omega_uniform=False,                              # set omegas uniformly random between set value and 0
                film_conditioning=False,                          # condition this layer using FiLM
                concat_conditioning=0,                            # condition this layer using concatenation
                init_method={"weights": 'basic', "bias": "zero"}) # weights: basic, kaiming_in, siren. bias: zero, polar
                :
                ...
    def forward(self, 
                layer_input, 
                z=None,     # for concatenation
                gamma=None, # for FiLM scaling
                beta=None,  # for FiLM shifting
                delta=None  # for post activation scaling
                ):
                ...
```

