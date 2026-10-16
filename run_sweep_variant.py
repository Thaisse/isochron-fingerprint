"""Re-run the sweep under altered numerical settings (decision-level
sensitivity).  usage: python run_sweep_variant.py TAG key=value ..."""
import json, sys, time
from multiprocessing import Pool
from isofp import core as c, sweep

TAG = sys.argv[1]
KW = {}
for a in sys.argv[2:]:
    k, v = a.split('=')
    KW[k] = int(v) if k == 'N' else (float(v) if k in ('half_width', 'n_periods', 'dmax', 'alpha') else v)

def job(k):
    fam, f, p, x0, tt = sweep.sweep_points()[k]
    d = c.run(f, p, x0, tt, **KW)
    return dict(family=fam, params=p, time={kk: float(v) for kk, v in d.items() if not kk.startswith('_')})

if __name__ == '__main__':
    with Pool(2) as pool:
        res = pool.map(job, range(len(sweep.sweep_points())), chunksize=1)
    json.dump(res, open(f'out/sweep_{TAG}.json', 'w'), indent=1)
    print('done', TAG)
