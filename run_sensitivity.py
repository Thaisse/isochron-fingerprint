"""Numerical stability (IEEE item 1) and sensitivity (item 2) on the five
representative models -> out/sensitivity.json"""
import json, sys, time, numpy as np
from multiprocessing import Pool
from isofp import core as c, models as m
from isofp.reps import REPS

KEYS = ['D1', 'CV', 'G', 'rc', 'nLT', 'kDC', 'k2', 'logR']

def settings():
    S = [('ref', {})]
    S += [(f'N={n}', dict(N=n)) for n in (40, 56, 96, 128, 160)]
    S += [(f'dt x{k}', dict(dtf=k)) for k in (0.25, 0.5, 2.0)]
    S += [(f'Tf={k}T', dict(n_periods=k)) for k in (2.0, 3.0, 8.0, 12.0)]
    S += [('Euler', dict(method='euler')), ('Euler dt x0.25', dict(method='euler', dtf=0.25))]
    S += [(f'dmax={k}', dict(dmax=k)) for k in (0.05, 0.20)]
    S += [(f'alpha={k}', dict(alpha=k)) for k in (1.5, 3.0)]
    S += [(f'window={k}', dict(half_width=k)) for k in (0.6, 1.2)]
    S += [(f'M={k}', dict(M=k)) for k in (600, 4800)]
    rng = np.random.default_rng(1)
    S += [(f'jitter{j}', dict(offset=tuple(rng.uniform(-0.5, 0.5, 2)))) for j in range(16)]
    return S

def job(ij):
    i, j = ij
    name, f, p, x0, tt = REPS[i]
    sname, kw = settings()[j]
    kw = dict(kw)
    M = kw.pop('M', 1200)
    cyc = c.limit_cycle(f, p, x0, tt, M=M)
    if 'dtf' in kw:
        kw['dt'] = kw.pop('dtf') * min(0.02, cyc.T / 400)
    t = time.time()
    d = c.run(f, p, None, None, cyc=cyc, **kw)
    return name, sname, {k: float(v) for k, v in d.items() if not k.startswith('_')}, time.time() - t

if __name__ == '__main__':
    jobs = [(i, j) for i in range(len(REPS)) for j in range(len(settings()))]
    out = {}
    with Pool(2) as pool:
        for name, sname, d, dt in pool.imap_unordered(job, jobs):
            out.setdefault(name, {})[sname] = d
            print(name, sname, '%.1fs' % dt, flush=True)
    json.dump(out, open('out/sensitivity.json', 'w'), indent=1)
