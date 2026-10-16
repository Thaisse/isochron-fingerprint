"""Parameter sweep (63 points, five families) -- Table S1 of the paper."""
import numpy as np
from . import models as m

def sweep_points():
    pts = []
    for tau in (10.0, 12.5, 15.0):
        for I in np.linspace(0.45, 1.32, 6):
            p = dict(m.FHN_BASE, I=float(I), tau=tau)
            pts.append(('FHN', m.fhn, p, [0.0, 0.0], 30 * 40))
    for I in np.linspace(95, 130, 6):
        p = dict(m.ML_TYPE2, I=float(I))
        pts.append(('ML-II', m.ml, p, [0.0, 0.3], 3000))
    for gCa in np.linspace(4.0, 4.8, 5):
        p = dict(m.ML_TYPE2, I=110.0, gCa=float(gCa))
        pts.append(('ML-II', m.ml, p, [0.0, 0.3], 3000))
    for I in np.linspace(45, 75, 7):
        p = dict(m.ML_TYPE1, I=float(I))
        pts.append(('ML-I', m.ml, p, [0.0, 0.3], 3000))
    for zb in (1.6, 1.9, 2.2):
        for I in np.linspace(2.2, 3.5, 4):
            p = dict(m.HR_BASE, I=float(I), zbar=zb)
            pts.append(('HR', m.hr_fast, p, [1.0, -1.0], 400))
    for mu in (0.8, 1.1, 1.4):
        for beta in np.linspace(0, 2, 5):
            p = dict(m.SL_BASE, mu=mu, beta=float(beta))
            pts.append(('SL', m.sl, p, [0.5, 0.1], 100))
    return pts
