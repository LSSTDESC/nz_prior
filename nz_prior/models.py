import numpy as np
from scipy.interpolate import interp1d


def shift_model(nz, shift):
    """
    Aplies a shift to the given p(z) distribution.
    This is done by interpolating the p(z) distribution
    at the shifted z values and then evaluating it at the
    original z values.
    """
    z, nz = nz
    nz_i = interp1d(z, nz, kind="linear", fill_value="extrapolate")
    pdf = nz_i(z + shift)
    norm = np.sum(pdf)
    return [z, pdf / norm]


def shift_and_width_model(nz, params):
    """
    Aplies a shift and a width to the given p(z) distribution.
    This is done by evluating the n(z) distribution at
    p((z-mu)/width + mu + shift) where mu is the mean redshift
    of the fiducial n(z) distribution and the rescaling by the width.
    Finally the distribution is normalized.
    """
    shift, width = params
    z, nz = nz
    nz_i = interp1d(z, nz, kind="linear", fill_value="extrapolate")
    mu = np.average(z, weights=nz)
    pdf = nz_i((z - mu + shift) / width + mu)
    norm = np.sum(pdf)
    return [z, pdf / norm]


def linear_model(nz_mean, W, alpha):
    """
    Linear model for the n(z) distribution.
    This is done by applying the linear transformation
    nz_mean + W * alpha where W is a matrix of weights
    and alpha is a vector of coefficients.
    """
    return nz_mean + np.dot(W, alpha)
