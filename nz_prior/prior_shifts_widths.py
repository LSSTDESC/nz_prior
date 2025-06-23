import numpy as np
from numpy.linalg import cholesky
from .prior_base import PriorBase
from .utils import make_cov_posdef


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
        self.sys_shift = self._find_sys_shift()
        self.sys_width = self._find_sys_width()

    def _find_shifts(self):
        mu = np.average(self.z, weights=self.nz_mean)
        shifts = [
            (np.average(self.z, weights=nz) - mu) for nz in self.nzs
        ]  # mean of each nz
        return shifts

    def _find_sys_shift(self):
        mu = np.average(self.z, weights=self.nz_mean)
        _mu = np.average(self.z, weights=self.nz_fid)
        sys_shift = _mu - mu
        return sys_shift

    def _find_widths(self):
        stds = []
        mean_mu = np.average(self.z, weights=self.nz_mean)
        mean_std = np.sqrt(np.average((self.z - mean_mu) ** 2, weights=self.nz_mean))
        for nz in self.nzs:
            mu = np.average(self.z, weights=nz)
            std = np.sqrt(np.average((self.z - mu) ** 2, weights=nz))
            stds.append(std)
        stds = np.array(stds)
        widths = stds / mean_std
        return widths

    def _find_sys_width(self):
        mean_mu = np.average(self.z, weights=self.nz_mean)
        mean_std = np.sqrt(np.average((self.z - mean_mu) ** 2, weights=self.nz_mean))
        fid_mu = np.average(self.z, weights=self.nz_fid)
        fid_std = np.sqrt(np.average((self.z - fid_mu) ** 2, weights=self.nz_fid))
        sys_width = fid_std / mean_std
        return sys_width

    def _get_prior(self):
        params = self._get_params().T
        sys_params = self._get_sys_params()
        mean = np.mean(params, axis=0)
        mean = 0.5 * (mean + sys_params)
        cov = np.cov(params, rowvar=False)
        cov += np.diag(sys_params**2)
        cov = make_cov_posdef(cov)
        chol = cholesky(cov)
        self.prior_mean = mean
        self.prior_cov = cov
        self.prior_chol = chol

    def _get_params(self):
        return np.array([self.shifts, self.widths])

    def _get_sys_params(self):
        return np.array([self.sys_shift, self.sys_width])

    def _get_params_names(self):
        return ["delta_z", "width_z"]
