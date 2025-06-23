import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import cholesky
from .prior_base import PriorBase


class PriorShifts(PriorBase):
    """
    Projector for the shifts model.
    The shift model assumes that all the variation in the measured
    photometric distributions can be described by a single shift in
    the position of the mean of a fiducial n(z) distribution.

    This shift is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The shift prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    """
    def __init__(self, ens, zgrid=None):
        self._prior_base(ens, zgrid=zgrid)
        self._find_prior()

    def _find_prior(self):
        self.shifts = self._find_shifts()
        self.sys_shift = self._find_sys_shift()

    def _find_shifts(self):
        mu = np.average(self.z, weights=self.nz_mean)
        shifts = [(np.average(self.z, weights=nz)-mu) for nz in self.nzs]
        return shifts

    def _find_sys_shift(self):
        mu = np.average(self.z, weights=self.nz_mean)
        _mu = np.average(self.z, weights=self.nz_fid)
        sys_shift = _mu - mu
        return sys_shift

    def _get_prior(self):
        shifts = self.shifts
        sys_shift = self.sys_shift
        mean = np.array([np.mean(shifts)])
        mean = 0.5 * (mean + sys_shift)
        s_shifts = np.std(shifts)
        cov = np.array([[s_shifts**2+sys_shift**2]])
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def _get_params(self):
        return np.array([self.shifts])

    def _get_sys_params(self):
        return np.array([self.sys_shift])

    def _get_params_names(self):
        return np.array(['delta_z'])
