import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from isofp import stats as S
res = json.load(open('out/sweep.json'))
fam, X = S.table(res, 'time')
lab = {'G': 'Gini $G$', 'CV': r'CV$(|\nabla\phi|)$', 'rc': '$r_c/D$',
       'D1': r'$\overline{|\nabla\phi|}D$', 'nLT': r'$-\Lambda T$',
       'kDC': r'$\kappa_{\mathrm{DC}}$ (PRC DC)', 'k2': r'$\kappa_2$ (PRC 2nd harm.)',
       'logR': r'$\log_{10}R_\Gamma$'}
order = ['kDC', 'logR', 'CV', 'k2', 'rc', 'D1', 'G', 'nLT']
order = sorted(order, key=lambda k: -S.eta2(X[k], fam))
cols = dict(zip(S.FAMS, ['C0', 'C1', 'C2', 'C3', 'C4']))
fig, ax = plt.subplots(2, 4, figsize=(15, 6.6))
rng = np.random.default_rng(0)
for a, k in zip(ax.ravel(), order):
    for i, f in enumerate(S.FAMS):
        v = X[k][fam == f]
        a.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=22, c=cols[f],
                  edgecolor='k', lw=.4, alpha=.85)
        a.plot([i - .25, i + .25], [v.mean()] * 2, 'k', lw=1.6)
    a.set_xticks(range(5)); a.set_xticklabels(['FHN', 'ML-II', 'ML-I', 'HR fast', 'SL'], rotation=30, fontsize=9)
    a.set_title(f'{lab[k]}\n$\\eta^2 = {S.eta2(X[k], fam):.2f}$', fontsize=10)
    a.grid(axis='y', alpha=.3)
    if k == 'nLT': a.set_yscale('log')
plt.tight_layout()
plt.savefig('figures/fig3_fingerprints.png', dpi=150)
print(order)
