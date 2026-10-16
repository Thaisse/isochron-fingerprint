"""Coordinate-transformation test (IEEE item 3) -> out/invariance.json

Each transformation is defined on standardised coordinates
u = (x - c)/e of the native limit cycle (c: centre, e: extent) and mapped
back, so its strength is comparable across models.  The transformed field
is f'(x') = D Phi(x) f(x) with x = Phi^{-1}(x').
"""
import json, numpy as np
from isofp import core as c, models as m

def cubic_inv(y, a):          # solve u + a u^3 = y (monotone), Cardano
    p, q = 1.0 / (3 * a), np.asarray(y, float) / (2 * a)
    s = np.sqrt(q * q + p ** 3)
    return np.cbrt(q + s) + np.cbrt(q - s)

th = np.deg2rad(30); cs, sn = np.cos(th), np.sin(th)
# (name, forward psi(u1,u2), inverse, jacobian entries (a11,a12,a21,a22)(u1,u2))
PSI = {
 'identity':        (lambda a, b: (a, b), lambda a, b: (a, b),
                     lambda a, b: (1, 0, 0, 1)),
 'shear (w+0.5v)':  (lambda a, b: (a, b + 0.5 * a), lambda a, b: (a, b - 0.5 * a),
                     lambda a, b: (1, 0, 0.5, 1)),
 'rotation 30deg':  (lambda a, b: (cs * a - sn * b, sn * a + cs * b),
                     lambda a, b: (cs * a + sn * b, -sn * a + cs * b),
                     lambda a, b: (cs, -sn, sn, cs)),
 'mixing (w+0.3v^2)': (lambda a, b: (a, b + 0.3 * a ** 2), lambda a, b: (a, b - 0.3 * a ** 2),
                     lambda a, b: (1, 0, 0.6 * a, 1)),
 'componentwise w+0.3w^3': (lambda a, b: (a, b + 0.3 * b ** 3), lambda a, b: (a, cubic_inv(b, 0.3)),
                     lambda a, b: (1, 0, 0, 1 + 0.9 * b ** 2)),
 'componentwise v+0.3v^3': (lambda a, b: (a + 0.3 * a ** 3, b), lambda a, b: (cubic_inv(a, 0.3), b),
                     lambda a, b: (1 + 0.9 * a ** 2, 0, 0, 1)),
}

def transformed(f, cen, ext, psi, psii, jac, scale=(1, 1), shift=(0, 0), tscale=1.0):
    scale = np.asarray(scale, float); shift = np.asarray(shift, float)
    def g(x, y, p):
        # x' = shift + scale * (cen + ext * psi((x - cen)/ext))
        a = (x - shift[0]) / scale[0]; b = (y - shift[1]) / scale[1]
        u1, u2 = psii((a - cen[0]) / ext[0], (b - cen[1]) / ext[1])
        X1, X2 = cen[0] + ext[0] * u1, cen[1] + ext[1] * u2
        f1, f2 = f(X1, X2, p)
        q1, q2 = f1 / ext[0], f2 / ext[1]
        j11, j12, j21, j22 = jac(u1, u2)
        r1, r2 = j11 * q1 + j12 * q2, j21 * q1 + j22 * q2
        return (scale[0] * ext[0] * r1 / tscale, scale[1] * ext[1] * r2 / tscale)
    def fwd(x):
        u1, u2 = psi((x[0] - cen[0]) / ext[0], (x[1] - cen[1]) / ext[1])
        return np.array([shift[0] + scale[0] * (cen[0] + ext[0] * u1),
                         shift[1] + scale[1] * (cen[1] + ext[1] * u2)])
    return g, fwd

MODELS = [('FHN', m.fhn, dict(m.FHN_BASE, I=0.972), [0., 0.], 1200),
          ('ML-II', m.ml, dict(m.ML_TYPE2, I=109.0), [0., 0.3], 3000),
          ('ML-I', m.ml, dict(m.ML_TYPE1, I=55.0), [0., 0.3], 3000)]
CASES = [('identity', 'identity', {}),
         ('translation', 'identity', dict(shift=(5.0, -3.0))),
         ('diag. scaling (v x10)', 'identity', dict(scale=(10, 1))),
         ('diag. scaling (w x100)', 'identity', dict(scale=(1, 100))),
         ('time rescaling (x10)', 'identity', dict(tscale=10.0)),
         ('componentwise w+0.3w^3', 'componentwise w+0.3w^3', {}),
         ('componentwise v+0.3v^3', 'componentwise v+0.3v^3', {}),
         ('shear (w+0.5v)', 'shear (w+0.5v)', {}),
         ('rotation 30deg', 'rotation 30deg', {}),
         ('mixing (w+0.3v^2)', 'mixing (w+0.3v^2)', {})]
KEYS = ['D1', 'CV', 'G', 'rc', 'nLT', 'kDC', 'k2', 'logR']

if __name__ == '__main__':
    out = {}
    for name, f, p, x0, tt in MODELS:
        base = c.limit_cycle(f, p, x0, tt)
        cen = 0.5 * (base.X.max(0) + base.X.min(0)); ext = 0.5 * (base.X.max(0) - base.X.min(0))
        for cname, pk, kw in CASES:
            psi, psii, jac = PSI[pk]
            g, fwd = transformed(f, cen, ext, psi, psii, jac, **kw)
            tts = tt * kw.get('tscale', 1.0)
            cyc = c.limit_cycle(g, p, fwd(base.X[0]), tts)
            r = c.run(g, p, None, None, cyc=cyc, labelling=('time', 'arc'))
            out[f'{name}|{cname}'] = {lab: {k: float(v) for k, v in d.items()
                                      if not k.startswith('_')} for lab, d in r.items()}
            d = out[f'{name}|{cname}']['time']
            print(name, cname, {k: round(d[k], 4) for k in KEYS},
                  'arcG=%.4f arcD1=%.4f' % (out[f'{name}|{cname}']['arc']['G'],
                                         out[f'{name}|{cname}']['arc']['D1']), flush=True)
    json.dump(out, open('out/invariance.json', 'w'), indent=1)
