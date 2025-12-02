import numpy as np
import matplotlib.pyplot as plt

def risk_neutral_density_St(x_array, cf, S0, r, q, T, hp,
                                u_max=200, n_steps=5000):
  
 

    # integration grid in u
    u = np.linspace(0.0, u_max, n_steps)    # shape (N_u,)
    # to make it into logSt char function
    phi = np.exp(1j * u * np.log(S0)) * cf(u, S0, T, r, q, hp)               # shape (N_u,)


    # shape (N_u, N_x): outer product in exponent
    exp_term = np.exp(-1j * np.outer(u, x_array))
    integrand = np.real(exp_term * phi[:, None])      # broadcast phi over columns

    # integrate over u (axis=0) to get f_X(x) for each x
    fx = (1/np.pi) * np.trapezoid(integrand, u, axis=0)
    return fx   # shape (N_x,)


def plot_risk_neutral_density_St(s_min, s_max, cf, S0, r, q, T, hp,
                                 u_max=200, n_steps=5000):
    """
    Plots f_{S_T}(s) and returns the density values.
    """
    s_array = np.linspace(s_min, s_max, 200)
    x_array = np.log(s_array)

    # f_X(x)
    fx = risk_neutral_density_St(x_array, cf, S0, r, q, T, hp, u_max=u_max, n_steps=n_steps)
    fS = fx / s_array

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_array, fS, color = 'black')
    ax.axvline(x=S0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$f_{S_T}(s)$")
    ax.set_title("Risk-neutral density of $S_T$")
    ax.grid(True)
    plt.show()

    


