"""Emit LaTeX table bodies from the JSON outputs."""
import json, numpy as np, sys
from isofp import stats as S
res = json.load(open('out/sweep.json'))
fam, X = S.table(res, 'time')
NM = {'FHN': 'FHN', 'ML-II': 'ML (Type II)', 'ML-I': 'ML (Type I)', 'HR': 'HR fast', 'SL': 'Stuart--Landau'}
fmt = {'D1': 3, 'CV': 3, 'G': 3, 'rc': 3, 'nLT': 2, 'kDC': 3, 'k2': 3}
L = []
for f in S.FAMS:
    m = fam == f
    cells = []
    for k, d in fmt.items():
        v = X[k][m]
        cells.append(f'${v.mean():.{d}f}\\!\\pm\\!{v.std(ddof=1):.{d}f}$')
    L.append(f'{NM[f]:15s} & {m.sum():2d} & ' + ' & '.join(cells) + r' \\')
out = {'fingerprint': '\n'.join(L)}
L = []
lab = {'D1': r'$\overline{|\nabla\phi|}D$', 'CV': 'CV', 'G': '$G$', 'rc': '$r_c/D$', 'nLT': r'$-\Lambda T$',
       'kDC': r'$\kappa_{\mathrm{DC}}$', 'k2': r'$\kappa_2$', 'logR': r'$\log_{10}R_\Gamma$'}
for k in S.ALL:
    a, b = X[k][fam == 'FHN'], X[k][fam == 'ML-II']
    d = 1 if k == 'nLT' else 3
    auc = S.auc(X[k], fam)
    e2 = S.eta2(X[k], fam)
    A = f'\\mathbf{{{auc:.3f}}}' if auc > 0.999 else f'{auc:.3f}'
    L.append(f'{lab[k]:28s} & ${e2:.3f}$ & $[{a.min():.{d}f},{a.max():.{d}f}]$ & $[{b.min():.{d}f},{b.max():.{d}f}]$ & ${A}$ \\\\')
out['sep'] = '\n'.join(L)
cl = {}
for name, fs in [('metric', S.METRIC), ('invar', S.INVAR), ('all', S.ALL)]:
    cl[name] = S.classify(res, 'time', fs, n_perm=300)
out['clf'] = {k: [float(x) for x in v] for k, v in cl.items()}
# variants (decision-level sensitivity)
var = {}
for tag in ['Euler', 'N48', 'N112', 'W12']:
    try:
        r2 = json.load(open(f'out/sweep_{tag}.json'))
    except FileNotFoundError:
        continue
    f2, X2 = S.table(r2, 'time')
    var[tag] = dict(acc=S.classify(r2, 'time', S.ALL, n_perm=100)[0],
                    acc_metric=S.classify(r2, 'time', S.METRIC, n_perm=100)[0],
                    auc={k: S.auc(X2[k], f2) for k in S.ALL},
                    eta2={k: S.eta2(X2[k], f2) for k in S.ALL})
out['variants'] = var
json.dump(out, open('out/tables.json', 'w'), indent=1)
print(out['fingerprint']); print(out['sep']); print(out['clf'])
for t, v in var.items():
    print(t, 'acc %.3f metric %.3f' % (v['acc'], v['acc_metric']), {k: round(x, 3) for k, x in v['auc'].items()})
    print('   eta2', {k: round(x, 3) for k, x in v['eta2'].items()})
