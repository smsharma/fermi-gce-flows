# Simulation-based inference for the Galactic Center Excess

[Siddharth Mishra-Sharma](smsharma@mit.edu) and [Kyle Cranmer](kyle.cranmer@nyu.edu)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-2110.xxxxx%20-green.svg)](https://arxiv.org/abs/2110.xxxxx)

![Summary of model.](paper/arXiv-v1/plots/hig.png)

## Abstract

The nature of the Fermi gamma-ray Galactic Center Excess (GCE) has remained a persistent mystery for over a decade. Although the excess is broadly compatible with emission expected due to dark matter annihilation, an explanation in terms of a population of unresolved astrophysical point sources e.g., millisecond pulsars, remains viable. The effort to uncover the origin of the GCE is hampered in particular by an incomplete understanding of diffuse emission of Galactic origin. This can lead to spurious features that make it difficult to robustly differentiate smooth emission, as expected for a dark matter origin, from more "clumpy" emission expected for a population of relatively bright, unresolved point sources. We use recent advancements in the field of simulation-based inference, in particular density estimation techniques using normalizing flows, in order to characterize the contribution of modeled components, including unresolved point sources, to the GCE. Compared to traditional techniques based on the statistical distribution of photon counts, our machine learning-based method is able to utilize more of the information contained in a given model of the Galactic Center emission, and in particular can perform posterior parameter estimation while accounting for pixel-to-pixel spatial correlations in the gamma-ray map. On application to Fermi data, the method generically attributes a smaller fraction of the GCE flux to unresolved point source-like emission when compared to traditional approaches. We nevertheless infer such a contribution to make up a non-negligible fraction of the GCE across all analysis variations considered, with at least 38<sup>+9</sup><sub>-19</sub>% of the excess being consistent with a point source origin in our baseline analysis.

## Code

**Note:** This code uses a custom version of PyGSP, which can be installed as follows:
```
git clone https://github.com/smsharma/pygsp.git -b sphere-graphs
cd pygsp
python setup.py install
```

## Citation

```
@article{Mishra-Sharma:2021abc,
      author         = "Mishra-Sharma, Siddharth and Cranmer, Kyle",
      title          = "{A neural simulation-based inference approach for characterizing the Galactic Center $\gamma$-ray excess}",
      year           = "2021",
      eprint         = "2110.xxxxx",
      archivePrefix  = "arXiv",
      primaryClass   = "astro-ph.HE",
      SLACcitation   = "%%CITATION = ARXIV:2110.xxxxx;%%"
}
```

The repository contains 
- Code that is part of [sbi](https://github.com/mackelab/sbi) for inference,
- [Code](https://github.com/smsharma/mining-for-substructure-lens) associated with [1909.02005](https://arxiv.org/abs/1909.02005) for scripting and data processing, 
- Code that is part of [NPTFit-Sim](https://github.com/nickrodd/NPTFit-Sim) for forward modeling, and
- [Code](https://github.com/deepsphere/deepsphere-pytorch) associated with [2012.15000](https://arxiv.org/abs/2012.15000) for constructing the feature extractor network.