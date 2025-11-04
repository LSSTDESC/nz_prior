import numpy as np
from scipy.interpolate import interp1d


def shift_and_width_model(z, nz, shift, width):
    """
    Aplies a shift and a width to the given p(z) distribution.
    This is done by evluating the n(z) distribution at
    p((z-mu)/width + mu + shift) where mu is the mean redshift
    of the fiducial n(z) distribution and the rescaling by the width.
    Finally the distribution is normalized.
    """
    nz_i = interp1d(z, nz, kind="linear", fill_value="extrapolate")
    mu = np.average(z, weights=nz)
    pdf = nz_i((z - mu - shift) / width + mu)
    norm = np.sum(pdf)
    return pdf / norm


def linear_model(nz, W, alpha):
    """
    Linear model for the n(z) distribution.
    This is done by applying the linear transformation
    nz_mean + W * alpha where W is a matrix of weights
    and alpha is a vector of coefficients.
    """
    return nz + np.dot(W, alpha)
