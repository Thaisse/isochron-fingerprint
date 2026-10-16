"""Stuart-Landau ground truth (Tables I-II) -> out/validation.json"""
import json, numpy as np
from isofp import core as c, models as m

p = dict(m.SL_BASE, mu=1.0, beta=1.0)
cyc = c.limit_cycle(m.sl, p, [.5, .1], 100)
N = 80
g1, g2, G1, G2, h = c.make_grid(cyc, N)
X1, X2 = G1 / cyc.S[0], G2 / cyc.S[1]
r = np.hypot(X1, X2)
ann = (r > 0.35) & (r < 1.6)
# phase origin: exact phase of the cycle point labelled 0
off = m.sl_exact_phase(cyc.X[0, 0], cyc.X[0, 1], p)
exact = np.mod(m.sl_exact_phase(X1, X2, p) - off, 2 * np.pi)
out = {}
for name, meth, dt, nper in [('RK4 dt=T/400, 4T (reference)', 'rk4', min(0.02, cyc.T / 400), 4),
                             ('Euler dt=0.02, 4T', 'euler', 0.02, 4),
                             ('Euler dt=0.08, 3T', 'euler', 0.08, 3),
                             ('RK4 dt=0.01, 4T', 'rk4', 0.01, 4),
                             ('RK4 dt=0.02, 60T', 'rk4', 0.02, 60)]:
    A, B, te = c.integrate_grid(cyc, G1, G2, nper, dt, meth)
    th = c.assign_phase(cyc, A, B, te)
    err = np.degrees(np.abs(np.angle(np.exp(1j * (th - exact)))))[ann]
    out[name] = dict(median=float(np.nanmedian(err)), p95=float(np.nanpercentile(err, 95)))
    print(name, out[name])
# gradient conventions (reference protocol: RK4, dt=min(0.02,T/400), 4T)
A, B, te = c.integrate_grid(cyc, G1, G2, 4, min(0.02, cyc.T / 400), 'rk4')
th = c.assign_phase(cyc, A, B, te)
Zr = c.adjoint_prc(cyc) / cyc.S
rn = np.hypot(G1, G2)                                  # nondim radius
ex_mag = np.sqrt(1 + p['beta'] ** 2) / rn              # exact |grad| (nondim)
def summ(mag):
    v = mag[np.isfinite(mag)]
    return dict(D1=float(v.mean() * cyc.D), CV=float(v.std() / v.mean()), G=float(c.gini(v)))
gw = c.phase_gradient(th, h, True); gn = c.phase_gradient(th, h, False)
out['grad'] = {'exact': summ(ex_mag), 'wrap': summ(np.hypot(*gw)), 'naive': summ(np.hypot(*gn))}
# relative error of |grad| on annulus
rel = np.abs(np.hypot(*gw) - ex_mag) / ex_mag
out['grad_relerr_median'] = float(np.nanmedian(rel[ann]))
# full descriptors vs exact-field descriptors on same grid
dnum = c.descriptors(cyc, g1, g2, G1, G2, th, *gw, Z_ref=Zr)
ex1 = -(np.sqrt(1 + 1) / rn) * 0  # placeholder not used
# exact gradient components in nondim coordinates
x, y = G1, G2
r2 = x * x + y * y
ex_g1 = (-y + p['beta'] * x) / r2
ex_g2 = (x + p['beta'] * y) / r2
dex = c.descriptors(cyc, g1, g2, G1, G2, exact, ex_g1, ex_g2, Z_ref=Zr)
out['desc_numeric'] = {k: float(v) for k, v in dnum.items() if not k.startswith('_')}
out['desc_exactfield'] = {k: float(v) for k, v in dex.items() if not k.startswith('_')}
print(out['grad'], out['grad_relerr_median'])
for k in ['D1', 'CV', 'G', 'rc', 'kDC', 'k2', 'logR']:
    print(k, out['desc_numeric'][k], out['desc_exactfield'][k])
json.dump(out, open('out/validation.json', 'w'), indent=1)
