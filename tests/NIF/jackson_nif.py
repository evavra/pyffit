import time
import numpy as np
from scipy.linalg import cholesky, qr, solve_triangular
import matplotlib.pyplot as plt

def sqrt_update(H, R, y, x_f, P_f, get_S_c=False):
    """
    DESCRIPTION:
    Perform Kalman filter update step using factored formulation.
    Based on NIF implementation available from Noel Jackson (KU) and the Stanford group.

    REFERENCES:
    Bartlow, N. M., S. Miyazaki, A. M. Bradley, and P. Segall (2011), Space-time correlation of
        slip and tremor during the 2009 Cascadia slow slip event, Geophys. Res. Lett., 38, L18309, 
        doi:10.1029/2011GL048714.

    Miyazaki, S., P. Segall, J. J. McGuire, T. Kato, and Y. Hatanaka (2006), Spatial and tmporal 
        evolution of stress and slip rate during the 2000 Tokai slow earthquake, J. Geophys. Res., 
        111, B03409, doi:10.1029/2004JB003426

    Segall, P., and M. Matthews (1997), Time dependent inversion of geodetic data, J. Geophys. 
        Res., 102, 22,391-22,409.

    INPUT:
    H     = the state -> observation matrix.
    R_c   = chol(R), where R is the observation covariance matrix.
    y     = the vector of observations.
    x_f   = x_k|k-1 (forecasted state)
    P_f   = chol(P_k|k-1) (forecasted covariance)

    OUTPUT:
    x_a   = x_k|k (analyzed state)
    P_a   = P_k|k (analyzed covariance)
    z     = y - H*x_f
    S_c   = chol(H P_k|k-1 H' + R) (if get_S_c=True).

    NOTES (from Paul Segall):
    z and S_c can be used to calculate the likelihood
        p(y_k | Y_k-1),

    where Y_k-1(:,i) is measurement vector i, as follows. First,
        p(y_k | Y_k-1) = N(y_k; H x, R + H P_k|k-1 H')
                       = N(y_k - Hx; 0, S_c*S_c')

    where N(x; mu, Sigma) is the density for a normal distribution having mean mu and covariance Sigma. 
    Second, if R = chol(A), then
        log(det(A)) = 2*sum(log(diag(R))).

    Hence,
        log p(y_k | Y_k-1) = -n/2 log(2 pi) - sum(log(diag(S_c))) - 1/2 q q',

    where q = z' / S_c (or q = (S_c' \ z)'), n = length(y_k), and we have just taken the logarithm of the 
    expression for the normal probability density function.

    For
        A = [chol(R)           0
             chol(P_k|k-1) H'  chol(P_k|k-1)],

    the Schur complement of (A'A)(1:ny, 1:ny) in A'A is
       P_k|k = P_k|k-1 - P_k|k-1 H' inv(S) H P_k|k-1,

    where
        S = H P_k|k-1 H' + R.

    The key idea in this square-root filter is that
       [~, R] = qr(A)

    is a safe operation in the presence of numerical error and so assures a
    factorization of the filtered covariance matrix P_k|k.
    """
    
    n, m = H.shape

    # Perform Cholesky factorizations
    R_c   = cholesky(R, lower=False)
    P_f_c = cholesky(P_f, lower=False)
    zeros = np.zeros((n, m))
    C     = P_f_c @ H.T 

    # Form block matrix A
    A = np.block([[R_c, zeros], 
                  [C,   P_f_c]])

    # Perform QR decomposition on A
    P_a_c = qr(A, mode='r')[0]

    # Extract the part of chol(A) that is chol(P_k|k),
    # I.e., the factorization of the Schur complement of interest.
    P_a_c          = P_a_c[n:n+m, n:n+m]
    mask           = np.diag(P_a_c < 0)
    P_a_c[mask, :] = -P_a_c[mask, :]

    # Innovation
    z = y - H @ x_f

    # Filtered state
    tmp1 = solve_triangular(R_c, z, lower=True)
    tmp2 = solve_triangular(R_c.T, tmp1, lower=False)
    tmp3 = H.T @ tmp2
    tmp4 = P_a_c @ tmp3
    x_a  = x_f + P_a_c.T @ tmp4

    # Get full P_c
    P_a = P_a_c.T @ P_a_c

    if get_S_c:
        # Extract chol(S).
        S_c          = P_a_c[:n, :n]
        mask         = np.diag(S_c < 0)
        S_c[mask, :] = -S_c[mask, :]
        return x_a, P_a, z, S_c
    
    else:
        return x_a, P_a, z
