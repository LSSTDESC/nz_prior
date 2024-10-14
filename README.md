# n(z) prior

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/LSSTDESC/rail-prior/branch/main/graph/badge.svg)](https://codecov.io/gh/LSSTDESC/rail-prior)
[![PyPI](https://img.shields.io/pypi/v/rail_prior?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/rail_prior/)

A DESC Package to turn an ensemble of redshift distributions into a prior for photo-z uncertainty.

![image](https://github.com/user-attachments/assets/db975890-934e-4db1-aefd-51f6709aac29)

## Usage

`nz_prior` can either be run on a `QP` ensemble or on a `sacc` file with `QPnz` tracers. Here is an example of the simplest use when a `QP` ensemble is provided:

```python
import nz_prior as nzp
X = nzp.PriorX(q) 
#  X is the method to capture photometric uncertainty
#  q is the sacc file with QPnz tracers
```

Here is an example when a `sacc` file is provided instead:

```python
X = nzp.PriorSacc(sacc_file, 
    model="Model",
    compute_crosscorrs="Full",
    zgrid=zgrid)
```
A couple of running options are important to note:
- model: the model used to capture the uncertainty in the n(z)'s within the `sacc` file.
    - "Shifts":  the n(z)'s are shifted by a constant amount.
    - "ShiftsWidths": the n(z)'s are shifted by a constant amount and broadened by a constant width.
    - "PCA": the n(z)'s are represented as a linear combination of the principal components.
    - "Fourier": the n(z)'s are represented as a linear combination of Fourier modes.
    - "GP": the model assumes that the n(z)'s are a Gaussian Process.
    - "Comb": the n(z)'s are represented as a linear combination of evenly spaced Gaussians.
- compute_crosscorrs: whether to compute the cross-correlations between the n(z)'s or not.
    - "None": do not compute the cross-correlations.
    - "BinWise": compute the cross-correlations between the parameters associated with the same tomographic bin.
    - "Full": compute the cross-correlations between all the parameters across tomographic bins.
- zgrid: the zgrid on which to evaluiate the n(z)'s. If None, the zgrid of the `sacc` file is used.

Upon initializing the class, Gaussian priors for the model parameters will be calibrated. The prior can then be accessed as an attribute of the class:

```python
mean, cov, chol = X.get_prior()
```
which returns the mean, covariance, and Cholesky decomposition of the prior.

Moreover, it is possible to sample from the prior:

```python
samples = X.sample_prior()
```
which procudes a dictionary with the following structure `Dict{param_names: value}`. The keys are the names of the parameters and the values are the samples.

The samples can then be used to generate n(z)'s using the models in the `nz_prior.models` module.

Checkout the examples folder for more detailed use cases.

## Installation

To install the package, run the following command in the root directory of the package:

```bash
pip install .
```
Note: make sure that you have installed sacc version 0.16 or higher.
