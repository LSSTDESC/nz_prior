import numpy as np
from numpy.linalg import eig, cholesky


def make_cov_posdef(cov):
    if not is_pos_def(cov):
        print('Warning: Covariance matrix is not positive definite')
        print('The covariance matrix will be regularized')
        jitter = 1e-15 * np.eye(cov.shape[0])
        w, v = eig(cov+jitter)
        w = np.real(np.abs(w))
        v = np.real(v)
        cov = v @ np.diag(np.abs(w)) @ v.T
        cov = np.tril(cov) + np.triu(cov.T, 1)
        if not is_pos_def(cov):
            print('Warning: regularization failed')
            print('The covariance matrix will be diagonalized')
            cov = np.diag(np.diag(cov))+jitter
    return cov

def is_pos_def(A):
    try:
        cholesky(A)
        return True
    except np.linalg.linalg.LinAlgError as err:
        return False

def wiener_filter(z_new, z_old, nzs):
    N, M = len(z_old), len(z_new)
    concat_nzs = []
    for nz in nzs:
        nz_new = np.interp(z_new, z_old, nz)
        concat_nz = np.append(nz_new, nz)
        concat_nzs.append(concat_nz)
    concat_cov = np.cov(np.array(concat_nzs).T)
    Koo = concat_cov[M:, M:]
    Koo_inv = np.linalg.pinv(Koo)
    Kon = concat_cov[:M, M:]
    C = Kon @ Koo_inv
    new_samples = np.array([C @ nz for nz in nzs])
    return new_samples

def Dkl(mu_1, K_1, mu_2, K_2):
    r = mu_1 - mu_2
    K_1 = make_cov_posdef(K_1)
    K_2 = make_cov_posdef(K_2)
    K_2_inv = np.linalg.pinv(K_2)
    K_1_inv = np.linalg.pinv(K_1)
    K_1_det = np.linalg.det((10**8)*K_1)
    K_2_det = np.linalg.det((10**8)*K_2)
    T1 = np.trace(K_2_inv @ K_1)
    T2 = r @ K_1_inv @ r
    T3 = np.log(K_2_det / K_1_det)
    T4 = len(mu_1)
    return 0.5 * (T1 + T2 - T3 - T4)

def Sym_Dkl(mu_1, K_1, mu_2, K_2):
    K_1 = make_cov_posdef(K_1)
    K_2 = make_cov_posdef(K_2)
    Dkl_12 = Dkl(mu_1, K_1, mu_2, K_2)
    Dkl_21 = Dkl(mu_2, K_2, mu_1, K_1)
    return 0.5 * (Dkl_12 + Dkl_21)
