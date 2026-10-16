import json, numpy as np, sys
from isofp import stats as S
res = json.load(open(sys.argv[1] if len(sys.argv)>1 else 'out/sweep.json'))
for lab in ['arc','time']:
    fam, X = S.table(res, lab)
    print('=====', lab)
    for f in S.FAMS:
        mm = fam == f
        print(f'{f:6s} n={mm.sum():2d} ' + ' '.join(f'{k}={X[k][mm].mean():.3f}±{X[k][mm].std(ddof=1):.3f}' for k in S.ALL) + f' phaseless={X["phaseless_frac"][mm].mean():.3f}')
    for k in S.ALL:
        m1, m2 = fam=='FHN', fam=='ML-II'
        print(f'  {k:5s} eta2={S.eta2(X[k],fam):.3f} AUC={S.auc(X[k],fam):.3f} FHN[{X[k][m1].min():.3f},{X[k][m1].max():.3f}] ML2[{X[k][m2].min():.3f},{X[k][m2].max():.3f}]')
    for name, fs in [('metric',S.METRIC),('invar',S.INVAR),('all',S.ALL)]:
        print('  clf', name, S.classify(res, lab, fs, n_perm=int(sys.argv[2]) if len(sys.argv)>2 else 100))
