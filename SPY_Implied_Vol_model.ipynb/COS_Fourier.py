import numpy as np

# cf is a characteristic function of Levy Process for St
# p are the parameters of the Levy process
def European_Call(cf, S0, K, T, r, q, p, N = 2**15, L = 15):
    
    a = 0 - L * np.sqrt(T)
    b = 0 + L * np.sqrt(T)
    x0 = np.log(S0/K)
    k = np.arange(N)                      # 0,1,...,N-1
    u = k * np.pi / (b - a)               # frequencies

   


    # Compute Vk coefficients analytically

    c = 0
    d = b                                

    

    # psi_k(c,d)
    psi = np.zeros(N)
    psi[0] = d - c
    psi[1:] = (np.sin(k[1:] * np.pi * (d - a)/(b - a)) - np.sin(k[1:] * np.pi * (c - a)/(b - a))) * (b - a) / (k[1:] * np.pi)

    # chi_k(c,d)
    term1 = 1.0 / (1.0 + (k * np.pi / (b - a)) ** 2)
    term2 = np.cos(k * np.pi  * ((d - a)/ (b - a))) * np.exp(d)
    term3 = np.cos(k * np.pi  * ((c - a)/ (b - a))) * np.exp(c)
    term4 = (k * np.pi / (b - a)) * np.sin(k * np.pi * ((d - a)/(b - a))) * np.exp(d)
    term5 = (k * np.pi / (b - a)) * np.sin(k * np.pi * ((c - a)/(b - a))) * np.exp(c)
    chi = term1 * (term2  - term3 + term4 - term5)

    Vk = 2.0 / (b - a) * (chi - psi)
    Vk[0] *= 0.5  # first term has weight 1/2 in COS expansion

    # Characteristic function part
    temp = (cf(u, S0, T, r , q, p) * Vk)
    mat = np.exp(1j * (x0 - a) * u)
    value = np.exp(-r * T) * K * np.real(mat @ temp)
 
    
    return float(value)

# via Call - Put parity
def European_Put(cf, S0, K, T, r, q, p, N = 2**15, L = 15):
    C = European_Call(cf, S0, K, T, r, q, p, N = 2**15, L = 15)
    return C - S0*np.exp(-q*T) + K*np.exp(-r*T)

