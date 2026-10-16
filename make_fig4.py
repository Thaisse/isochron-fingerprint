"""Sensitivity/stability figure + tables from out/sensitivity.json"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = json.load(open('out/sensitivity.json'))
K = ['D1', 'CV', 'G', 'rc', 'nLT', 'kDC', 'k2', 'logR']
KL = [r'$\overline{|\nabla\phi|}D$', 'CV', '$G$', '$r_c/D$', r'$-\Lambda T$',
      r'$\kappa_{\rm DC}$', r'$\kappa_2$', r'$\log R_\Gamma$']
MOD = ['FHN', 'ML-II', 'ML-I', 'HR', 'SL']
GROUPS = [('integrator: Euler vs RK4', ['Euler', 'Euler dt x0.25']),
          (r'step $\Delta t$ (0.25$\times$–2$\times$)', ['dt x0.25', 'dt x0.5', 'dt x2.0']),
          ('horizon $T_f$ (2$T$–12$T$)', ['Tf=2.0T', 'Tf=3.0T', 'Tf=8.0T', 'Tf=12.0T']),
          ('cycle samples $M$ (600–4800)', ['M=600', 'M=4800']),
          ('grid offset (16 sub-cell shifts)', [f'jitter{j}' for j in range(16)]),
          ('grid $N$ = 56 / 96', ['N=56', 'N=96']),
          ('grid $N$ (40–160)', ['N=40', 'N=56', 'N=96', 'N=128', 'N=160']),
          ('threshold $d_{\\max}$ (0.05–0.2)', ['dmax=0.05', 'dmax=0.2']),
          (r'threshold $\alpha$ (1.5–3)', ['alpha=1.5', 'alpha=3.0']),
          ('window (0.6$D$–1.2$D$)', ['window=0.6', 'window=1.2'])]

def rel(m, s, k):
    a, b = R[m][s][k], R[m]['ref'][k]
    if abs(b) < 1e-6:
        return 0.0 if abs(a) < 1e-6 else np.inf
    return 100 * abs(a - b) / abs(b)

H = np.zeros((len(GROUPS), len(K))); Hmed = np.zeros_like(H)
for i, (_, ss) in enumerate(GROUPS):
    for j, k in enumerate(K):
        v = [rel(m, s, k) for m in MOD for s in ss]
        H[i, j] = np.max(v)
        # per-model max, then median over models
        Hmed[i, j] = np.median([max(rel(m, s, k) for s in ss) for m in MOD])
# jitter: report CV across offsets (incl. ref) as stability measure
jit = {}
for m in MOD:
    for k in K:
        v = np.array([R[m][f'jitter{j}'][k] for j in range(16)] + [R[m]['ref'][k]])
        jit[(m, k)] = 100 * v.std() / abs(v.mean()) if abs(v.mean()) > 1e-6 else 0.0
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.35, 1]))
Hp = np.clip(np.where(np.isfinite(Hmed), Hmed, 1e3), 0.01, 1e3)
im = ax[0].imshow(np.log10(Hp), cmap='RdYlGn_r', vmin=-1, vmax=2.3, aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = Hmed[i, j]
        ax[0].text(j, i, ('%.1f' % v) if v < 100 else '%.0f' % v, ha='center', va='center', fontsize=8)
ax[0].set_xticks(range(len(K))); ax[0].set_xticklabels(KL, fontsize=9)
ax[0].set_yticks(range(len(GROUPS))); ax[0].set_yticklabels([g[0] for g in GROUPS], fontsize=9)
ax[0].set_title('(a) max. relative change (%) vs. reference; median over 5 models', fontsize=10)
cb = fig.colorbar(im, ax=ax[0], fraction=0.03); cb.set_label('log$_{10}$ %')
Ns = [40, 56, 72, 96, 128, 160]
for j, k in enumerate(K[:4]):
    curves = []
    for m in MOD:
        vals = [R[m]['ref' if n == 72 else f'N={n}'][k] for n in Ns]
        ref = vals[-1]
        if abs(ref) < 1e-6: continue
        curves.append([100 * abs(v - ref) / abs(ref) for v in vals[:-1]])
    med = np.median(np.array(curves), 0)
    ax[1].plot(Ns[:-1], np.maximum(med, 1e-3), 'o-', label=KL[j], lw=1.4, ms=4)
ax[1].set_yscale('log'); ax[1].set_xlabel('grid resolution $N$')
ax[1].set_ylabel('|change| vs. $N=160$ (%), median over models')
ax[1].set_title('(b) grid convergence (grid-free descriptors: no change)', fontsize=10); ax[1].grid(alpha=.3)
ax[1].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig('figures/fig4_robustness.png', dpi=160)
json.dump(dict(groups=[g[0] for g in GROUPS], H_max=H.tolist(), H_med=Hmed.tolist(),
               jitter={f'{a}|{b}': v for (a, b), v in jit.items()}),
          open('out/sensitivity_summary.json', 'w'), indent=1)
print('rows', [g[0] for g in GROUPS]); print('cols', K)
np.set_printoptions(precision=2, suppress=True, linewidth=150)
print('MEDIAN over models\n', Hmed); print('MAX over models\n', H)
for m in MOD:
    print('jitter CV%', m, [round(jit[(m, k)], 3) for k in K])
