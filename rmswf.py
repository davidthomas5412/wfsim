import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from tqdm import tqdm

from wfTel import LSSTFactory

factory = LSSTFactory('i')

amplitudes = [300e-6]*2   # M2 x/y
amplitudes += [20e-6]     # M2 z
amplitudes += [30e-6]*2   # M2 tilt
amplitudes += [1e-3]*2    # camera x/y
amplitudes += [20e-6]     # camera z
amplitudes += [100e-6]*2  # camera tilt
amplitudes += [5e-7]*20   # M1M3 zers
amplitudes += [5e-7]*20   # M2 zers

titles = [f'M2 {dof}' for dof in ['x', 'y', 'z', 'thx', 'thy']]
titles += [f'camera {dof}' for dof in ['x', 'y', 'z', 'thx', 'thy']]
titles += [f'M1M3 zer {i+1}' for i in range(20)]
titles += [f'M2 zer {i+1}' for i in range(20)]

scales = [1e6]*3      # micron
scales += [206265]*2  # arcsec
scales += [1e6]*3     # micron
scales += [206265]*2  # arcsec
scales += [1e6]*40       # micron

units = ['micron']*3
units += ['arcsec']*2
units += ['micron']*3
units += ['arcsec']*2
units += ['micron']*40

fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(15, 20))

lefts = []
rights = []
for i, (amplitude, title, scale, unit) in enumerate(tqdm(zip(amplitudes, titles, scales, units))):
    perturbs = np.linspace(-amplitude, amplitude, 15)
    dzs = []
    for perturb in perturbs:
        visit = factory.make_visit_telescope(dof=[0]*i+[perturb]+[0]*(50-i))
        dzs.append(visit.dz())
    dzs = np.array(dzs)
    rms = np.sqrt(np.sum(dzs[:,:,4:]**2, axis=(1,2)))

    # Solve each side for rms = 0.3
    def resid(p, target):
        visit = factory.make_visit_telescope(dof=[0]*i+[p]+[0]*(50-i))
        return np.sqrt(np.sum(visit.dz()[:,4:]**2)) - target

    # left
    left = brentq(resid, -2*amplitude, 0, args=(0.3,))
    right = brentq(resid, 0, 2*amplitude, args=(0.3,))
    print(f"{title:15s}  {left:8.3g}   {right:8.3g}")
    lefts.append(left)
    rights.append(right)

    ax = axes.ravel()[i]
    ax.plot(perturbs*scale, rms)
    ax.set_xlabel(unit)
    ax.set_ylabel("rms wavefront")
    ax.set_title(title)
    ax.set_ylim(0.15, 0.5)
    ax.axvline(left*scale, c='r')
    ax.axvline(right*scale, c='r')


print()
print()
for title, left, right in zip(titles, lefts, rights):
    print(f"{title:15s}  {left:8.3g}   {right:8.3g}")

fig.tight_layout()
plt.savefig("rms.png")
# plt.show()
