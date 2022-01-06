import galsim
import numpy as np

idx = 0

rng = galsim.BaseDeviate(57721)

boresight = galsim.CelestialCoord(0 * galsim.degrees, 0 * galsim.degrees)
q1 = 0 * galsim.degrees
cq1, sq1 = np.cos(q1), np.sin(q1)
affine1 = galsim.AffineTransform(cq1, -sq1, sq1, cq1)
radecToField1 = galsim.TanWCS(
    affine1,
    boresight,
    units=galsim.radians
)
ra1 = boresight.ra + 1 * galsim.degrees
dec1 = boresight.dec + 1.5 * galsim.degrees
coord1 = galsim.CelestialCoord(ra1, dec1)

q2 = 270 * galsim.degrees
cq2, sq2 = np.cos(q2), np.sin(q2)
affine2 = galsim.AffineTransform(cq2, -sq2, sq2, cq2)
radecToField2 = galsim.TanWCS(
    affine2,
    boresight,
    units=galsim.radians
)
ra2 = boresight.ra + (np.cos(q2) + 1.5 * np.sin(q2)) * galsim.degrees
dec2 = boresight.dec + (-np.sin(q2) + 1.5 * np.cos(q2) ) * galsim.degrees
coord2 = galsim.CelestialCoord(ra2, dec2)

field1 = radecToField1.toImage(coord1)
field2 = radecToField2.toImage(coord2)

print("cord 1 ", coord1)
print("cord 2 ", coord2)

print("field 1 ", field1)
print("field 2 ", field2)

print("field 1 pix", (field1.x - field2.x) / 10e-6)