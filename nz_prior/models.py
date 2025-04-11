import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


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
    pdf = nz_i((z - mu + shift) * width + mu)
    norm = np.sum(pdf)
    return [z, pdf / norm]


def comb_model(nz, W):
    ncombs = len(W)
    z, nz = nz
    dz = (np.max(z) - np.min(z)) / ncombs
    zmeans = [(np.min(z) + dz / 2) + i * dz for i in range(ncombs)]
    combs = {}
    for i in np.arange(ncombs):
        combs[i] = norm(zmeans[i], dz / 2)
    nz_pred = np.zeros(len(z))
    for i in np.arange(ncombs):
        nz_pred += (W[i] / ncombs) * combs[i].pdf(z)
    return [z, nz_pred / np.sum(nz_pred)]


def pca_model(nz, W, eigvecs):
    z, nz = nz
    new_nz = nz + np.dot(eigvecs.T, W)
    return [z, new_nz]


def gp_model(nq, nz_mean, W):
    z, nz_mean = nz_mean
    new_nz = nz_mean + np.dot(W, nq)
    return [z, new_nz]


def linear_model(nz_mean, W):
    _, n = W.shape
    alpha = np.random.normal(size=n)
    return nz_mean + np.dot(W, alpha)


def fourier_model(nz, W):
    z, nz = nz
    n = len(W)
    WW = np.zeros(len(z), dtype=complex)
    WW[:n] = W
    new_nz = np.fft.ifft(WW)
    return [z, nz + new_nz]
