from lsst.obs.lsst import LsstCamMapper
from lsst.afw.cameraGeom.cameraSys import FOCAL_PLANE, FIELD_ANGLE
"""Extract and pickle chip coordinates from LSST-DM software.
   Re-ran on 4/4/20, now includes wavefront sensors.
"""

camera = LsstCamMapper().camera
out = dict()
for det in camera:
    name = det.getName()
    corners_pos = [[c.x * 1e-3, c.y * 1e-3] for c in det.getCorners(FOCAL_PLANE)]
    center_pos = det.getCenter(FOCAL_PLANE)
    corners_field = [[c.x, c.y] for c in det.getCorners(FIELD_ANGLE)]
    center_field = det.getCenter(FIELD_ANGLE)
    out[name] = {
        'center_pos': [center_pos.x * 1e-3, center_pos.y * 1e-3],
        'corners_pos': corners_pos,
        'center_field': [center_field.x, center_field.y],
        'corners_field': corners_field
    }

import pickle
with open("chips.pkl", 'wb') as f:
    pickle.dump(out, f)