import numpy as np
from scipy.interpolate import interp1d
from numpy.linalg import cholesky
from .prior_base import PriorBase


class PriorShiftsWidths(PriorBase):
    """
    Prior for the shifts and widths model.
    The shifts and widths model assumes that the variation in the measured
    photometric distributions can be captured by varying the mean and the
    standard deviation of a fiducial n(z) distribution.

    The calibration method was written by Tilman Tröster.
    The shift prior is given by a Gaussian distributiob with zero mean
    standard deviation the standard deviation in the mean of
    the measured photometric distributions.
    The width is calibrated by computing the standard deviations
    of the measured photometric distributions over redshift.
    The width prior is then given by a Gaussian distribution with
    mean 0 and variance equal to the ratio of the standard deviation
    of the standard deviations to the mean of the standard deviations.
    This is similar to how the shift prior is calibrated in the shift model.
    """
    def __init__(self, ens, zgrid=None):
        self._prior_base(ens, zgrid=zgrid)
        self._find_prior()

    def _find_prior(self):
        self.shifts = self._find_shifts()
        self.widths = self._find_widths()

    def _find_shifts(self):
        mu = np.average(self.z, weights=self.nz_mean)
        shifts = [(np.average(self.z, weights=nz)-mu) for nz in self.nzs]   # mean of each nz
        return shifts

    def _find_widths(self):
        stds = []
        for nz in self.nzs:
            mu = np.average(self.z, weights=nz)
            std = np.sqrt(np.average((self.z-mu)**2, weights=nz))
            stds.append(std)
        stds = np.array(stds)
        std_mean = np.mean(stds)        # mean of the stds
        widths = stds / std_mean
        return widths

    def _get_prior(self):
        m_shift = np.mean(self.shifts)
        m_width = np.mean(self.widths)
        s_shift = np.std(self.shifts)
        s_width = np.std(self.widths)
        mean = np.array([m_shift, m_width])
        cov = np.array([
            [s_shift**2, 0],
            [0, s_width**2]])
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def _get_params(self):
        return np.array([self.shifts, self.widths])

    def _get_params_names(self):
        return ['delta_z', 'width_z']
