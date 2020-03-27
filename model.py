from collections import namedtuple
import numpy as np
from galsim.zernike import Zernike, zernikeRotMatrix

# http://stackoverflow.com/a/6849299
class lazy_property(object):
    """
    meant to be used for lazy evaluation of an object attribute.
    property should represent non-mutable data, as it replaces itself.
    """
    def __init__(self, fget):
        self.fget = fget
        self.func_name = fget.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.fget(obj)
        setattr(obj, self.func_name, value)
        return value


# Dict storing which Zernikes indices are fit together, singly,
# or are invalid (by their omission)
_nj = {
    4:1,
    5:2,
    7:2,
    9:2,
    11:1,
    12:2,
    14:2,
    16:2,
    18:2,
    20:2
}


DZ = namedtuple('DZ', ['k', 'j'])

class WFModelConfig:
    """Configuration for modeling wavefront as series of Double Zernike
    polynomials, including separate terms for the telescope, CCDs, and
    visit-level perturbations.

    Parameters
    ----------
    tel : {'lsst', None}
        Shortcut enabling appropriate degrees of freedom to model the static
        telescope contribution to the wavefront.  Specifying 'lsst' means to use
        a set of double Zernikes appropriate for the fiducial LSST optical
        design.  Default of None means to manually specify a double Zernike list
        below.
    visit : {'rigid', 'through10', 'through20', None}
        Shortcut enabling appropriate degrees of freedom for visit-to-visit
        modeling: rigid-body perturbations only, rigid-body + first 10 M1M3
        modes, or rigid-body + first 20 M1M3 modes.  Default of None means to
        manually specify a double Zernike list below.
    ccd : {'piston', 'tilt', None}
        Shortcut enabling appropriate degrees of freedom to model each ccd's
        contribution to the wavefront.  'piston' means allow chip height
        displacements only, 'tilt' means allow height and slope variations.  The
        default of None means to manually specify a double Zernike list below.
    telDZs : list of 2-tuples of ints, optional
        Each tuple indicates a (field, pupil) double Zernike to include in
        modeling the telescope contribution to the wavefront.  These terms are
        added to the terms indicated by the tel argument.
    visitDZs : list of 2-tuples of ints, optional
        Each tuple indicates a (field, pupil) double Zernike to include in
        modeling the visit-to-visit contribution to the wavefront.  These terms
        are added to the terms indicated by the visit argument.
    ccdDZs : list of 2-tuples of ints, optional
        Each tuple indicates a (field, pupil) double Zernike to include in
        modeling each CCD's contribution to the wavefront.  The "field" here is
        understood to be centered on each CCD with radius equal to the CCD
        diagonal.  These terms are added to the terms indicated by the ccd
        argument.
    zeroCCDSum : bool, optional
        Require CCD solutions to sum to zero?
    zeroVisitSum : bool, optional
        Require visit solutions to sum to zero?

    Notes
    -----

    The CCD term is defined in the CCD coordinate frame, whereas the
    telescope and visit terms are defined in the telescope coordinate
    frame.  This means we need to apply a rotation to the CCD
    coefficients when fitting or evaluating the wavefront model.
    """
    def __init__(
        self,
        tel=None, visit=None, ccd=None,
        telDZs=None, visitDZs=None, ccdDZs=None,
        zeroCCDSum=False, zeroVisitSum=False
    ):
        if tel is not None:
            if tel == 'lsst':
                self.telDZs = [
                    (13, 5), (12, 6),   # 0.1314 contribution to rms wfe
                    (9, 9), (10, 10),   # 0.0960
                    (5, 5), (6, 6),     # 0.0699
                    (1, 4),             # 0.0451
                    (4, 11),            # 0.0357
                    (11, 4),            # 0.0292
                    (8, 8), (7, 7),     # 0.0291
                    (1, 11),            # 0.0201
                    (10, 18), (9, 19),  # 0.0184
                    (4, 4),             # 0.0164
                    (1, 22),            # 0.0141
                    (6, 12), (5, 13),   # 0.0137
                    (5, 23), (6, 24),   # 0.0120
                    (12, 12), (13, 13), # 0.0110
                ]
            else:
                raise ValueError(f"Unknown telescope {tel}")
        else:
            self.telDZs = []

        if telDZs is not None:
            self.telDZs.extend(
                set(telDZs).difference(self.telDZs)
            )

        self.telDZs = [DZ(j, k) for j, k in self.telDZs]

        if visit is not None:
            if visit == 'rigid':
                self.visitDZs = [
                    (1, 7), (1, 8),  # constant coma
                    (2, 6), (3, 5),  # linear astigmatism
                    (2, 5), (3, 6),  # linear astigmatism
                    (2, 4), (3, 4),  # field tilt
                    (1, 4)           # focus
                ]
            elif visit == 'through10':
                self.visitDZs = [
                    (1, 7), (1, 8),     # constant coma
                    (2, 6), (3, 5),     # linear astigmatism
                    (2, 5), (3, 6),     # linear astigmatism
                    (2, 4), (3, 4),     # field tilt
                    (1, 4),             # focus

                    (1, 5), (1, 6),     # constant astig
                    (1, 13), (1, 12),   # constant second astig
                    (1, 9), (1, 10),    # constant trefoil
                    (1, 14), (1, 15),   # constant quatrefoil
                ]
            elif visit == 'through20':
                self.visitDZs = [
                    (1, 7), (1, 8),     # constant coma
                    (2, 6), (3, 5),     # linear astigmatism
                    (2, 5), (3, 6),     # linear astigmatism
                    (2, 4), (3, 4),     # field tilt
                    (1, 4),             # focus

                    (1, 5), (1, 6),     # constant astig
                    (1, 13), (1, 12),   # constant second astig
                    (1, 9), (1, 10),    # constant trefoil
                    (1, 14), (1, 15),   # constant quatrefoil

                    (2, 8), (3, 7),     # linear coma
                    (5, 5), (6, 6),     # astig astig
                    (1, 20), (1, 21),   # constant pentafoil
                    (3, 13), (2, 12),   # linear second astig
                    (6, 10), (5, 9),    # astig trefoil
                    (1, 18), (1, 19),   # const second trefoil
                    (3, 14), (2, 15),   # linear quatrefoil
                    (1, 27), (1, 28)    # constant hexafoil
                ]
            else:
                raise ValueError(f"Unknown visit dof {visit}")
        else:
            self.visitDZs = []

        if visitDZs is not None:
            self.visitDZs.extend(
                set(visitDZs).difference(self.visitDZs)
            )

        self.visitDZs = [DZ(j, k) for j, k in self.visitDZs]

        if ccd is not None:
            if ccd == 'piston':
                self.ccdDZs = [
                    (1, 4)
                ]
            elif ccd == 'tilt':
                self.ccdDZs = [
                    (1, 4),
                    (2, 4), (3, 4)
                ]
            else:
                raise ValueError(f"Unknown ccd dof {ccd}")
        else:
            self.ccdDZs = []

        if ccdDZs is not None:
            self.ccdDZs.extend(
                set(ccdDZs).difference(self.ccdDZs)
            )

        self.ccdDZs = [DZ(j, k) for j, k in self.ccdDZs]

        self.jmax = np.max([dz.j for dz in self.telDZs+self.visitDZs+self.ccdDZs])
        self.kmax = np.max([dz.k for dz in self.telDZs+self.visitDZs+self.ccdDZs])
        self.zeroCCDSum = zeroCCDSum
        self.zeroVisitSum = zeroVisitSum

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class WFDesignMatrix:
    """Class to hold the design matrix for the linear least squares fit
    to measure donut Zernike data.

    Parameters
    ----------
    config : WFModelConfig
        Configuration parameters for this fit.
    cat : (nDonut,) astropy.table.Table
        Table with at least the following columns
            x, y : float
                Telescope frame coordinates of donuts.
            visit : int
                Visit number for each donut.
            ccd : int
                CCD number for each donut.
            hra : float
                Horizontal rotation angle in radians.
            z4, ..., zj : float
                Columns for measured pupil Zernike coefficients.
        Note that the catalog should be sorted first by visit, then by ccd.
    j : int
        Starting Zernike index for which to construct design matrix.
        Note that some Zernike terms must be fit together, like z5
        and z6.  In this case j=5 will work and j=6 will raise an
        Exception.  Either one or two Zernikes will be fit together.
    focalRadius : float, optional
        Radius to use for Zernikes defined over the focal plane.
        Default: 1.0.
    ccdRadius : float, optional
        Radius to use for Zernikes defined over a single CCD.  Default: 1.0.
    """
    def __init__(
        self, config, cat, j,
        focalRadius=1.0, ccdRadius=1.0
    ):
        from galsim.zernike import noll_to_zern
        self.config = config
        self.cat = cat
        self.j = j
        self.nDonut = len(cat)
        self.ccds = np.unique(cat['ccd'])
        self.visits = np.unique(cat['visit'])

        self.focalRadius = focalRadius
        self.ccdRadius = ccdRadius

        n, m = noll_to_zern(self.j)
        if m == 0:
            self.single = True
            self.js = [self.j]
        else:
            n1, m1 = noll_to_zern(self.j+1)
            if n1 != n or m1 != -m:
                raise ValueError("Invalid j")
            self.single = False
            self.js = [self.j, self.j+1]

        self.ncol = len(self.columns)
        self.nrow = len(self.rows)

    @lazy_property
    def columns(self):
        """
        Notes
        -----
        Columns are organized hierarchically as:
        - j in {j, j+1}
          - kTel in kTels
          - visit in visits
            - kvisit in kVisits
          - ccd in ccds
            - kccd in kCCDs
        """
        cfg = self.config
        cat = self.cat

        # col idx -> col name
        #   col name could be 'tel', DZ
        #   or 'visit', visit, DZ
        #   or 'ccd', ccd, DZ
        cols = []

        # inverse of cols
        # col name -> col idx
        self.indexDict = {}

        # inverse of telescope part of cols
        # DZ -> idx
        self.telDict = {}

        # inverse of visit part of cols
        # visit, DZ -> idx
        self.visitDict = {}

        # inverse of ccd part of cols
        # ccd, DZ -> idx
        self.ccdDict = {}

        idx = 0
        for j in self.js:
            telKs = np.sort([dz.k for dz in cfg.telDZs if dz.j == j])
            visitKs = np.sort([dz.k for dz in cfg.visitDZs if dz.j == j])
            # Assume for now that CCD terms only exist if j=4, and they're
            # already appropriately paired.
            ccdKs = np.sort([dz.k for dz in cfg.ccdDZs if dz.j == j])

            for k in telKs:
                entry = DZ(k, j)
                cols.append(('tel', entry))
                self.indexDict['tel', entry] = idx
                self.telDict[entry] = idx
                idx += 1
            for visit in np.unique(cat['visit']):
                for k in visitKs:
                    entry = visit, DZ(k, j)
                    cols.append(('visit', entry))
                    self.indexDict['visit', entry] = idx
                    self.visitDict[entry] = idx
                    idx += 1
            for ccd in self.ccds:
                for k in ccdKs:
                    entry = ccd, DZ(k, j)
                    cols.append(('ccd', entry))
                    self.indexDict['ccd', entry] = idx
                    self.ccdDict[entry] = idx
                    idx += 1
        return cols

    @lazy_property
    def rows(self):
        """
        Notes
        -----
        Rows are organized hierarchically as:
        - j in {j, j+1}
          - visit in visits
            - ccd in ccds
              - donut in donuts
          - zeroCCDSum constraint
          - zeroVisitSum constraint
        """
        cfg = self.config
        cat = self.cat

        # row idx -> row name
        #   row name could be 'donut', (j, catIdx)
        #   or 'zeroVisitSum', DZ
        #   or 'zeroCCDSum', DZ
        rows = []
        # RHS in design matrix eqn
        self.constraints = []

        # (j, catIdx) -> row index
        self.rowDict = {}

        # j -> row slice
        self.telRowDict = {}

        # j, visit -> row slice
        self.visitRowDict = {}

        # j, visit, ccd -> row slice
        self.ccdRowDict = {}

        # DZ -> row slice
        self.zeroVisitRowDict = {}

        # DZ -> row slice
        self.zeroCCDRowDict = {}

        idx = 0
        for j in self.js:
            for catIdx, donut in enumerate(cat):
                entry = j, catIdx
                rows.append(('donut', entry))
                self.rowDict[entry] = idx
                self.constraints.append(donut[f'z{j}'])
                idx += 1
            visitKs = np.sort([dz.k for dz in cfg.visitDZs if dz.j == j])
            ccdKs = np.sort([dz.k for dz in cfg.ccdDZs if dz.j == j])
            if cfg.zeroVisitSum:
                # For every k, sum of visit terms is forced to 0.0
                for k in visitKs:
                    entry = DZ(k, j)
                    rows.append(('zeroVisitSum', entry))
                    self.rowDict['zeroVisitSum', entry] = idx
                    self.constraints.append(0.0)
            if cfg.zeroCCDSum:
                # For every k, sum of ccd terms is forced to 0.0
                for k in ccdKs:
                    entry = DZ(k, j)
                    rows.append(('zeroCCDSum', entry))
                    self.rowDict['zeroCCDSum', entry] = idx
                    self.constraints.append(0)

        # Go back and determine blocks
        offset = 0
        for j in self.js:
            visitKs = np.sort([dz.k for dz in cfg.visitDZs if dz.j == j])
            ccdKs = np.sort([dz.k for dz in cfg.ccdDZs if dz.j == j])

            self.telRowDict[j] = slice(offset, offset + len(cat))
            offset += len(cat)
            if cfg.zeroVisitSum:
                for k in visitKs:
                    self.zeroVisitRowDict[DZ(k, j)] = slice(offset, offset + 1)
                    offset += 1
            if cfg.zeroCCDSum:
                for k in ccdKs:
                    self.zeroCCDRowDict[DZ(k, j)] = slice(offset, offset + 1)
                    offset += 1
        offset = 0
        for j in self.js:
            for visit in np.unique(cat['visit']):
                nVisitDonut = np.sum(cat['visit'] == visit)
                self.visitRowDict[j, visit] = slice(
                    offset,
                    offset + nVisitDonut
                )
                offset += nVisitDonut
            if cfg.zeroVisitSum:
                offset += len(visitKs)
            if cfg.zeroCCDSum:
                offset += len(ccdKs)
        offset = 0
        for j in self.js:
            for visit in np.unique(cat['visit']):
                for ccd in self.ccds:
                    nVisitCCDDonut = np.sum(
                        (cat['visit'] == visit) &
                        (cat['ccd'] == ccd)
                    )
                    self.ccdRowDict[j,visit,ccd] = slice(
                        offset,
                        offset + nVisitCCDDonut
                    )
                    offset += nVisitCCDDonut
            if cfg.zeroVisitSum:
                offset += len(visitKs)
            if cfg.zeroCCDSum:
                offset += len(ccdKs)
        return rows

    @lazy_property
    def design(self):
        design = np.zeros((self.nrow, self.ncol), dtype=float)
        self._fillTelDesign(design)
        self._fillVisitDesign(design)
        self._fillCCDDesign(design)
        return design

    def _fillTelDesign(self, design):
        for dz, icol in self.telDict.items():
            Z = Zernike([0]*dz.k+[1], R_outer=self.focalRadius)
            catSlice = self.telRowDict[self.j]
            designRowSlice = self.telRowDict[dz.j]
            cat = self.cat[catSlice]
            coefs = Z(cat['thx'], cat['thy'])
            design[designRowSlice, icol] = coefs

    def _fillVisitDesign(self, design):
        for (visit, dz), icol in self.visitDict.items():
            Z = Zernike([0]*dz.k+[1], R_outer=self.focalRadius)
            catSlice = self.visitRowDict[self.j, visit]
            designRowSlice = self.visitRowDict[dz.j, visit]
            cat = self.cat[catSlice]
            coefs = Z(cat['thx'], cat['thy'])
            design[designRowSlice, icol] = coefs
            if self.config.zeroVisitSum:
                design[self.zeroVisitRowDict[dz], icol] = 1.0

    def _fillCCDDesign(self, design):
        for (ccd, dz), icol in self.ccdDict.items():
            if self.single:
                Z = Zernike([0]*dz.k+[1], R_outer=self.ccdRadius)
                for visit in np.unique(self.cat['visit']):
                    catSlice = self.ccdRowDict[dz.j, visit, ccd]
                    designRowSlice = catSlice
                    cat = self.cat[catSlice]
                    coefs = Z(cat['ccdx'], cat['ccdy'])
                    design[designRowSlice, icol] = coefs
            else:
                raise NotImplementedError()
            if self.config.zeroCCDSum:
                design[self.zeroCCDRowDict[dz], icol] = 1.0


class WFFitResult:
    def __init__(self, config, design, result):
        self.config = config
        self.design = design
        self.result = result  # indexed like design.columns

    def getTelWF(self, cat):
        cfg = self.config
        design = self.design
        thx = np.atleast_1d(cat['thx'])
        thy = np.atleast_1d(cat['thy'])
        out = np.zeros((len(thx), cfg.jmax+1), dtype=float)

        for j in range(4, cfg.jmax+1):
            coefs = np.zeros(cfg.kmax+1)
            for dz, idx in design.telDict.items():
                if dz.j != j:
                    continue
                coefs[dz.k] = self.result[idx]
            Z = Zernike(coefs, R_outer=design.focalRadius)
            out[:, j] = Z(thx, thy)
        return out

    def getVisitWF(self, cat):
        cfg = self.config
        design = self.design
        thx = np.atleast_1d(cat['thx'])
        thy = np.atleast_1d(cat['thy'])
        out = np.zeros((len(thx), cfg.jmax+1), dtype=float)

        for visit in design.visits:
            w = np.nonzero(cat['visit'] == visit)[0]
            for j in range(4, cfg.jmax+1):
                coefs = np.zeros(cfg.kmax+1)
                for (visit_, dz), idx in design.visitDict.items():
                    if dz.j != j or visit != visit_:
                        continue
                    coefs[dz.k] = self.result[idx]
                Z = Zernike(coefs, R_outer=design.focalRadius)
                out[w, j] = Z(thx[w], thy[w])
        return out

    def getCCDWF(self, cat):
        cfg = self.config
        design = self.design
        ccdx = np.atleast_1d(cat['ccdx'])
        ccdy = np.atleast_1d(cat['ccdy'])
        out = np.zeros((len(ccdx), cfg.jmax+1), dtype=float)
        for ccd in design.ccds:
            w = np.nonzero(cat['ccd'] == ccd)[0]
            for j in range(4, cfg.jmax+1):
                coefs = np.zeros(cfg.kmax+1)
                for (ccd_, dz), idx in design.ccdDict.items():
                    if dz.j != j or ccd != ccd_:
                        continue
                    coefs[dz.k] = self.result[idx]
                Z = Zernike(coefs, R_outer=design.ccdRadius)
                out[w, j] = Z(ccdx[w], ccdy[w])
        for visit in design.visits:
            w = np.nonzero(cat['visit'] == visit)[0]
            rotation = cat[w]['rotation'][0]
            out[w] = np.dot(zernikeRotMatrix(cfg.jmax, rotation), out[w].T).T
        return out
