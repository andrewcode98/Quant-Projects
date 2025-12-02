from dataclasses import dataclass
import numpy as np

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


@dataclass
class VGBparams:
    beta: float
    eta: float
    sigma: float
    theta: float

@dataclass
class NIGparams:
    alpha : float
    beta : float
    delta: float
    sigma: float


@dataclass
class HestonParams:
    kappa: float
    sigma: float
    theta: float
    v0: float
    rho: float

# Characteristic function in log-strike

def simpson_weights(N: int):
    if N % 2 != 0:
        raise ValueError("N must be even")
    w = np.ones(N)
    w[1:N-1:2] = 4
    w[2:N-2:2] = 2
    return w
# alpha damping
# eta frequency step
def fft_calls(cf, S0: float, T: float, r:float, q:float, p, N:int = 4096, eta:float = 0.25, alpha:float = 1.5):
    # Will return (N,) ascending strikes
    # and (N,) call prices for those strikes
    n = np.arange(N)
    v = eta * n # frequency grid
    i = 1j
    phi_shift = cf(v - (alpha + 1)*i, S0, T, r, q, p)
    denom = (alpha**2 + alpha - v**2 + i*(2*alpha + 1)*v)
    psi = np.exp(-r*T) * phi_shift / denom

    # Simpson weights for the v-integral
    w = simpson_weights(N) * (eta / 3.0)

    # FFT coupling
    lam = 2.0 * np.pi / (N * eta)   # Δk (log-strike step)
    b   = 0.5 * N * lam             # half-width in k
    x   = psi * np.exp(1j * b * v) * w

    F   = np.fft.fft(x)
    F   = np.real(F)

    j = np.arange(N)
    k = -b + j * lam                 # k = ln K
    K = np.exp(k)

    calls = np.exp(-alpha * k) / np.pi * F
    order = np.argsort(K)
    return K[order], np.maximum(calls[order], 0.0)

def fft_call_price(
    cf, S0: float, K: float, T: float, r: float, q: float, p,
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
):
    """Price a single call via FFT + linear interpolation on the K-grid."""
    K_grid, C_grid = fft_calls(cf, S0, T, r, q, p, N=N, eta=eta, alpha=alpha)
    if K <= K_grid[0]:
        return C_grid[0]
    if K >= K_grid[-1]:
        return C_grid[-1]
    idx = np.searchsorted(K_grid, K)
    x0, x1 = K_grid[idx-1], K_grid[idx]
    y0, y1 = C_grid[idx-1], C_grid[idx]
    return y0 + (y1 - y0) * (K - x0) / (x1 - x0)

def heston_fft_put_price(
    cf, S0: float, K: float, T: float, r: float, q: float, p,
    N: int = 4096, eta: float = 0.25, alpha: float = 1.5
):
    """Put via put–call parity."""
    C = fft_call_price(cf, S0, K, T, r, q, p, N=N, eta=eta, alpha=alpha)
    return C - S0*np.exp(-q*T) + K*np.exp(-r*T)