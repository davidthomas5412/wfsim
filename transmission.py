import numpy as np

lsst = np.loadtxt('lsst_total_r.dat', delimiter=' ')
sdss = np.loadtxt('SLOAN_SDSS.r.dat', delimiter=' ')

def B(lam, T):
    c = 2.99792458e8  # speed of light in m/s
    k = 1.3806488e-23  # Boltzmann's constant J per Kelvin
    h = 6.62607015e-34  # Planck's constant in J s
    return (2 * h * c ** 2) / (lam ** 5) * 1 / (np.exp((h * c) / (lam * k * T)) - 1)

# int f(\lambda) * \lambda^{-2} d\lambda
S_lsst = lsst[:,1]
lam_lsst = lsst[:,0] * 1e-9
S_sdss = sdss[:,1]
lam_sdss = sdss[:,0] * 1e-10
n = 20
cache = np.zeros((n,2))
cache[:,0] = np.linspace(2000,12000,n) # temperatures
for i,T in enumerate(cache[:,0]):
    lsst_trans = 0.1 * np.sum(B(lam_lsst, T) * S_lsst * lam_lsst ** -1)
    sdss_trans = 2.5 * np.sum(B(lam_sdss, T) * S_sdss * lam_sdss ** -1)
    cache[i,1] = lsst_trans / sdss_trans

np.save('transmission_cache', cache)