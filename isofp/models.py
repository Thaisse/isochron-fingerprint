"""Planar neuronal models used in the isochron-fingerprint framework.

Each model is a function f(x, y, p) -> (dx, dy) that is vectorised over
NumPy arrays x, y.  Parameter dictionaries hold the full specification so
every number in the paper can be regenerated from this file.
"""
import numpy as np

# ---------------------------------------------------------------- FHN
FHN_BASE = dict(a=0.7, b=0.8, tau=12.5, I=1.0)

def fhn(x, y, p):
    return (x - x**3 / 3.0 - y + p['I'],
            (x + p['a'] - p['b'] * y) / p['tau'])

# ---------------------------------------------------------------- ML
# Rinzel & Ermentrout (1989) parameter sets
ML_TYPE2 = dict(C=20.0, gCa=4.4, gK=8.0, gL=2.0, VCa=120.0, VK=-84.0,
                VL=-60.0, V1=-1.2, V2=18.0, V3=2.0, V4=30.0, phi=0.04,
                I=100.0)
ML_TYPE1 = dict(C=20.0, gCa=4.0, gK=8.0, gL=2.0, VCa=120.0, VK=-84.0,
                VL=-60.0, V1=-1.2, V2=18.0, V3=12.0, V4=17.4,
                phi=1.0 / 15.0, I=45.0)

def ml(x, y, p):
    minf = 0.5 * (1 + np.tanh((x - p['V1']) / p['V2']))
    winf = 0.5 * (1 + np.tanh((x - p['V3']) / p['V4']))
    lam = np.cosh((x - p['V3']) / (2 * p['V4']))   # 1/tau_w
    dV = (p['I'] - p['gCa'] * minf * (x - p['VCa'])
          - p['gK'] * y * (x - p['VK']) - p['gL'] * (x - p['VL'])) / p['C']
    dw = p['phi'] * (winf - y) * lam
    return dV, dw

# ---------------------------------------------------------------- HR fast
HR_BASE = dict(a=1.0, b=3.0, c=1.0, d=5.0, I=2.0, zbar=1.914)

def hr_fast(x, y, p):
    return (y - p['a'] * x**3 + p['b'] * x**2 - p['zbar'] + p['I'],
            p['c'] - p['d'] * x**2 - y)

# ---------------------------------------------------------------- SL
SL_BASE = dict(mu=1.0, omega=1.0, beta=1.0)

def sl(x, y, p):
    r2 = x * x + y * y
    w = p['omega'] + p['beta'] * r2
    return (p['mu'] * x - r2 * x - w * y,
            p['mu'] * y - r2 * y + w * x)

def sl_exact_phase(x, y, p):
    r = np.hypot(x, y)
    return np.mod(np.arctan2(y, x) + p['beta'] * np.log(r / np.sqrt(p['mu'])),
                  2 * np.pi)

MODELS = dict(FHN=fhn, ML=ml, HR=hr_fast, SL=sl)
