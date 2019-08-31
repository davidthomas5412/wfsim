import os
import pickle

import numpy as np

import batoid
import galsim


NMODES = 20

def betterGridData(inputCoords, inputValue, outputGrid, k=20):
    from scipy.interpolate import griddata
    from sklearn.neighbors import KDTree

    output = griddata(inputCoords, inputValue, outputGrid,
                      method='cubic')
    tree = KDTree(inputCoords)
    _, neighbors = tree.query(outputGrid, k=k)
    toFix = np.nonzero(~np.isfinite(output))[0]
    for idx in toFix:
        neighbor = neighbors[idx]
        basis = galsim.zernike.zernikeBasis(
            10,
            inputCoords[neighbor, 0],
            inputCoords[neighbor, 1]
        )
        coefs, _, _, _ = np.linalg.lstsq(
            basis.T,
            inputValue[neighbor],
            rcond=None
        )
        value = galsim.zernike.Zernike(coefs).evalCartesian(
            *outputGrid[idx]
        )
        output[idx] = value
    return output

fiducial_telescope = batoid.Optic.fromYaml("LSST_r.yaml")

M1M3dir = "/Users/josh/src/M1M3_ML/data/"
M1M3fn = os.path.join(M1M3dir, "M1M3_1um_156_grid.txt")
M1M3data = np.loadtxt(M1M3fn).T

w1 = np.where(M1M3data[0] == 1)[0]
w3 = np.where(M1M3data[0] == 3)[0]
M1x = M1M3data[1][w1]
M1y = M1M3data[2][w1]
M3x = M1M3data[1][w3]
M3y = M1M3data[2][w3]
M1modes = M1M3data[3:3+NMODES, w1]
M3modes = M1M3data[3:3+NMODES, w3]

# Modes are currently in the direction normal to surface.
# We want perturbations in the direction of the optic axis.
# So divide by the z-component of the surface normal.
M1modes /= fiducial_telescope['LSST.M1'].surface.normal(M1x, M1y)[:,2]
M3modes /= fiducial_telescope['LSST.M3'].surface.normal(M3x, M3y)[:,2]
M1InputCoords = np.column_stack([M1x, M1y])
M3InputCoords = np.column_stack([M3x, M3y])

# Output grid exceeds the mirror extent by a few points to make interpolation sane.
xgrid = np.linspace(-4.4, 4.4, 101, endpoint=True)
xgrid, ygrid = np.meshgrid(xgrid, xgrid)
xgrid = xgrid.ravel()
ygrid = ygrid.ravel()
rgrid = np.hypot(xgrid, ygrid)
outputGrid = np.column_stack([xgrid, ygrid])

M1GriddedModes = np.empty((NMODES, len(outputGrid)), dtype=np.float)
for i, mode in enumerate(M1modes):
    print(i)
    M1GriddedModes[i] = betterGridData(
        M1InputCoords, mode, outputGrid
    )

M3GriddedModes = np.empty((NMODES, len(outputGrid)), dtype=np.float)
for i, mode in enumerate(M3modes):
    print(i)
    M3GriddedModes[i] = betterGridData(
        M3InputCoords, mode, outputGrid
    )

# Reshape for batoid API
xgrid = xgrid[:101]
ygrid = ygrid[::101]
M1GriddedModes = M1GriddedModes.reshape(20, 101, 101)
M3GriddedModes = M3GriddedModes.reshape(20, 101, 101)
with open("mirrorModes.pkl", 'wb') as f:
    pickle.dump(dict(M1Modes=M1GriddedModes, M3Modes=M3GriddedModes, xgrid=xgrid, ygrid=ygrid), f)
