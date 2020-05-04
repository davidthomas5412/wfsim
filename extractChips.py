from lsst.obs.lsst import LsstCamMapper
from lsst.afw.cameraGeom.cameraSys import FOCAL_PLANE
"""Extract and pickle chip coordinates from LSST-DM software.
   Re-ran on 4/4/20, now includes wavefront sensors.
"""

camera = LsstCamMapper().camera
out = dict()
for det in camera:
    name = det.getName()
    corners = [[c.x * 1e-3, c.y * 1e-3] for c in det.getCorners(FOCAL_PLANE)]
    center = det.getCenter(FOCAL_PLANE)
    out[name] = {
        'center': [center.x * 1e-3, center.y * 1e-3],
        'corners': corners,
    }

import pickle
with open("chips.pkl", 'wb') as f:
    pickle.dump(out, f)
