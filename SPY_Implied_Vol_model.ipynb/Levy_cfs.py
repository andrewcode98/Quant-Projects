import numpy as np
from dataclasses import dataclass

@dataclass
class SVJparams:
    kappa : float # reversion rate
    sigma: float # vol of vol
    theta : float # long term variance
    v0 : float # initial variance
    rho : float # corr coef of Brownian motion
    lamda: float # intensity of Poisson
    mu_j: float # mean of jump
    sigma_j: float # vol of jump

@dataclass
class HestonParams:
    kappa: float
    sigma: float
    theta: float
    v0: float
    rho: float

@dataclass
class KouParams:
    lamda: float
    sigma: float
    prob: float
    eta_plus: float
    eta_minus: float


# All characteristic functions of Levy processes for St

# SVJ
def SVJ_cf_St(u, S0, T, r, q, p: SVJparams):
    i = 1j
    a  = p.kappa * p.theta
    b  = p.kappa - p.rho * p.sigma * i * u
    d  = np.sqrt(b*b + (p.sigma**2) * (i*u + u*u))
    g  = (b - d) / (b + d)
    kappa_j = np.exp(p.mu_j + 0.5 * p.sigma_j ** 2) - 1.0
    lamda_kj = p.lamda * kappa_j

    eDT = np.exp(-d * T)
    one_minus_g_eDT = 1 - g * eDT
    one_minus_g     = 1 - g
    # small guards
    one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
    one_minus_g     = np.where(np.abs(one_minus_g)     < 1e-15, 1e-15, one_minus_g)

    # jump cf
    G = np.exp((p.lamda * T) * (np.exp(i * u * p.mu_j - 0.5 * p.sigma_j ** 2 * u ** 2) - 1))

    C = i*u*(r - q - lamda_kj)*T + (a/(p.sigma**2)) * ((b - d)*T - 2.0*np.log(one_minus_g_eDT/one_minus_g))
    D = ((b - d)/(p.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)
    return np.exp(C + D*p.v0) * G

# Heston
def Heston_cf_St(u, S0, T, r, q, p: HestonParams):
    """
    Characteristic function φ(u) = E[exp(i u S_T)] under Q.
    Uses the Little Heston Trap parametrization.
    u can be scalar or numpy array.
    """
    i = 1j
    a  = p.kappa * p.theta
    b  = p.kappa - p.rho * p.sigma * i * u
    d  = np.sqrt(b*b + (p.sigma**2) * (i*u + u*u))
    g  = (b - d) / (b + d)

    eDT = np.exp(-d * T)
    one_minus_g_eDT = 1 - g * eDT
    one_minus_g     = 1 - g
    # small guards
    one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
    one_minus_g     = np.where(np.abs(one_minus_g)     < 1e-15, 1e-15, one_minus_g)

    C = i*u*(r - q)*T + (a/(p.sigma**2)) * ((b - d)*T - 2.0*np.log(one_minus_g_eDT/one_minus_g))
    D = ((b - d)/(p.sigma**2)) * ((1 - eDT) / one_minus_g_eDT)
    return np.exp(C + D*p.v0)

# Kou-model
def Kou_cf_St(u, S0, T, r, q, p:KouParams):
    kappa = (p.prob * p.eta_plus) / (p.eta_plus - 1.0) + (1 - p.prob) * p.eta_minus / (p.eta_minus + 1.0) - 1.0

    M_Y = p.prob * p.eta_plus / (p.eta_plus - 1j * u) + (1 -p.prob) * p.eta_minus / (p.eta_minus + 1j * u)

    drift = (r - q - p.lamda * kappa - 0.5 * p.sigma ** 2) * T

    return np.exp(1j * u * drift - 0.5 * p.sigma ** 2 * u ** 2 * T + p.lamda * T * (M_Y - 1.0))