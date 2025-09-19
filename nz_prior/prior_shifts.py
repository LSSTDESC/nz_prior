import numpy as np
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
        super().__init__(ens, zgrid=zgrid)
        self.shifts = self._find_shifts()
        self.params = self._get_params()

    def _find_shifts(self):
        mu = np.average(self.z, weights=self.nz_mean)
        shifts = [
            (np.average(self.z, weights=nz) - mu) for nz in self.nzs
        ]  # mean of each nz
        return shifts

    def _get_prior(self):
        mean = np.array([np.mean(self.shifts)])
        cov = np.array([[np.std(self.shifts) ** 2]])
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol
        self.prior_transform = chol

    def _get_params(self):
        return np.array([self.shifts])

    def _get_params_names(self):
        return np.array(["delta_z"])
