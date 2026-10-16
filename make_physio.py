"""Fig. 5 and Table of descriptor <-> measurable-observable correspondence."""
import json, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from isofp import stats as S
from isofp.reps import REPS
import run_physio as RP

P = json.load(open('out/physio.json')); W = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'out/sweep.json'))
assert [p['family'] for p in P] == [w['family'] for w in W]
fam = np.array([p['family'] for p in P])
def X(k): return np.array([w['time'][k] for w in W])
def O(k): return np.array([p[k] for p in P])
def within(x, y):
    """Spearman correlation after ranking within each family (removes
    between-family differences)."""
    rx, ry = np.zeros(len(x)), np.zeros(len(y))
    for f in np.unique(fam):
        m = fam == f
        if np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0: rx[m] = ry[m] = np.nan; continue
        rx[m] = (rankdata(x[m]) - 1) / (m.sum() - 1) - .5
        ry[m] = (rankdata(y[m]) - 1) / (m.sum() - 1) - .5
    ok = np.isfinite(rx)
    return spearmanr(rx[ok], ry[ok])[0]
PAIRS = [('kDC', 'prc_kDC'), ('kDC', 'prc_adv_frac'), ('k2', 'prc_k2'),
         ('rc', 'eps_lin'), ('nLT', 'F2_over_F1'), ('G', 'F2_over_F1'),
         ('CV', 'F2_over_F1'), ('G', 'prc_gini'), ('phaseless_frac', 'annihilation_frac')]
res = {}
for d, o in PAIRS:
    r, p = spearmanr(X(d), O(o))
    res[f'{d}|{o}'] = dict(rho=float(r), p=float(p), rho_within=float(within(X(d), O(o))))
    print(f'{d:15s} {o:18s} rho={r:+.2f} p={p:.1e} within={res[f"{d}|{o}"]["rho_within"]:+.2f}')
summ = {f: {k: [float(np.mean(O(k)[fam == f])), float(np.std(O(k)[fam == f], ddof=1))]
            for k in ['prc_adv_frac', 'prc_kDC', 'eps_lin', 'F2_over_F1', 'annihilation_frac']}
        for f in S.FAMS}
json.dump(dict(pairs=res, family=summ), open('out/physio_summary.json', 'w'), indent=1)
for f in S.FAMS: print(f, {k: '%.3f±%.3f' % tuple(v) for k, v in summ[f].items()})

cols = dict(zip(S.FAMS, ['C0', 'C1', 'C2', 'C3', 'C4']))
names = {'FHN': 'FHN', 'ML-II': 'ML-II', 'ML-I': 'ML-I', 'HR': 'HR fast', 'SL': 'SL'}
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
# (a) measured PRCs
for name, f, p, x0, tt in REPS[:4]:
    cyc, r = RP.measure(f, p, x0, tt)
    base = r['dphi'][:, 0] / RP.EPS[0]
    ph = np.arange(RP.NPH) / RP.NPH
    ax[0].plot(ph, base / np.abs(base).max(), 'o-', ms=3, color=cols[name], label=names[name])
ax[0].axhline(0, color='k', lw=.6); ax[0].set_xlabel('stimulus phase $\\theta/2\\pi$')
ax[0].set_ylabel('phase shift (normalised)'); ax[0].legend(fontsize=8)
ax[0].set_title('(a) PRC from 1% current pulses', fontsize=10)
def sc(a, d, o, title, xl, yl, logy=False):
    for f in S.FAMS:
        m = fam == f
        a.scatter(X(d)[m], O(o)[m], s=20, c=cols[f], edgecolor='k', lw=.3, label=names[f])
    rr = res[f'{d}|{o}']
    a.set_title(f'{title}\n$\\rho = {rr["rho"]:+.2f}$ (within-family {rr["rho_within"]:+.2f})', fontsize=10)
    a.set_xlabel(xl); a.set_ylabel(yl)
    if logy: a.set_yscale('log')
    a.grid(alpha=.3)
sc(ax[1], 'kDC', 'prc_kDC', '(b) PRC type', r'$\kappa_{\rm DC}$ (adjoint)', 'DC index of pulse PRC')
ax[1].plot([0, .8], [0, .8], 'k--', lw=.7)
sc(ax[2], 'rc', 'eps_lin', '(c) linear-response range', '$r_c/D$', r'$\epsilon_{\rm lin}$ (fraction of $V_{pp}$)')
sc(ax[3], 'nLT', 'F2_over_F1', '(d) second-order PRC', r'$-\Lambda T$', r'rms$(F_2)$/rms$(F_1)$')
ax[3].set_xscale('log')
ax[1].legend(fontsize=7, loc='lower right')
plt.tight_layout(); plt.savefig('figures/fig5_physiology.png', dpi=160)
