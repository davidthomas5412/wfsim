import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCamMapper
import lsst.obs.lsst.cameraTransforms as cameraTransforms

"""Extract and pickle chip coordinates from LSST-DM software.
"""

camera = LsstCamMapper().camera
lct = cameraTransforms.LsstCameraTransforms(camera)
center = 509*4 + 0.5, 2000.5
out = {}
for det in camera:
    name = det.getName()
    x, y = lct.ccdPixelToFocalMm(*center, name)
    x *= 1e-3 # mm -> m
    y *= 1e-3
    print(f"{name}  {x:10.7f}  {y:10.7f}")
    left = x - 10e-6*(center[0])
    right = left + 10e-6*4072
    bottom = y - 10e-6*(center[1])
    top = bottom + 10e-6*4000
    out[name] = {
        'left':left,
        'right':right,
        'bottom':bottom,
        'top':top,
        'x':x,
        'y':y
    }

import pickle
with open("chips.pkl", 'wb') as f:
    pickle.dump(out, f)
