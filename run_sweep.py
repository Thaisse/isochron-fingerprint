"""Run the 63-point sweep with both phase labellings -> out/sweep.json"""
import json, sys, time, numpy as np
from multiprocessing import Pool
from isofp import core as c, sweep

def job(k):
    fam, f, p, x0, tt = sweep.sweep_points()[k]
    t = time.time()
    r = c.run(f, p, x0, tt, labelling=('time', 'arc'))
    out = {lab: {kk: float(v) for kk, v in d.items() if not kk.startswith('_')}
           for lab, d in r.items()}
    print(k, fam, '%.1fs' % (time.time() - t), flush=True)
    return dict(family=fam, params=p, **out)

if __name__ == '__main__':
    n = len(sweep.sweep_points())
    with Pool(2) as pool:
        res = pool.map(job, range(n), chunksize=1)
    json.dump(res, open('out/sweep.json', 'w'), indent=1)
