"""Representative parameter point of each family (centre of its sweep)."""
from . import models as m
REPS = [('FHN', m.fhn, dict(m.FHN_BASE, I=0.972, tau=12.5), [0., 0.], 1200),
        ('ML-II', m.ml, dict(m.ML_TYPE2, I=109.0), [0., 0.3], 3000),
        ('ML-I', m.ml, dict(m.ML_TYPE1, I=60.0), [0., 0.3], 3000),
        ('HR', m.hr_fast, dict(m.HR_BASE, I=2.6333, zbar=1.9), [1., -1.], 400),
        ('SL', m.sl, dict(m.SL_BASE, mu=1.1, beta=1.0), [.5, .1], 100)]
