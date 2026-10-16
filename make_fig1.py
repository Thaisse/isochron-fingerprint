"""Fig. 1: isochron foliations and phase-sensitivity maps (time-uniform
asymptotic phase, nondimensional coordinates)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from isofp import core as c, models as m

PAN = [('FHN', m.fhn, dict(m.FHN_BASE, I=0.972), [0., 0.], 1200),
       ('ML (Type II)', m.ml, dict(m.ML_TYPE2, I=109.0), [0., .3], 3000),
       ('ML (Type I)', m.ml, dict(m.ML_TYPE1, I=60.0), [0., .3], 3000),
       ('HR fast', m.hr_fast, dict(m.HR_BASE, I=2.2, zbar=1.9), [1., -1.], 400)]
fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
for k, (name, f, p, x0, tt) in enumerate(PAN):
    d, a = c.run(f, p, x0, tt, N=200, return_all=True)
    cyc, G1, G2, th = a['cyc'], a['G1'], a['G2'], a['th']
    mag = np.hypot(a['gr1'], a['gr2'])
    E = np.exp(1j * th)
    A0 = ax[0, k]
    for lev in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        Z = np.angle(E * np.exp(-1j * lev))
        Zm = np.where(np.abs(Z) < 2.5, Z, np.nan)       # avoid branch cut
        A0.contour(G1, G2, Zm, levels=[0], colors=[plt.cm.viridis(lev / 2 / np.pi)],
                   linewidths=0.9)
    A0.plot(*np.vstack([cyc.Y, cyc.Y[:1]]).T, 'r', lw=2)
    if np.isnan(th).any():
        A0.contourf(G1, G2, np.isnan(th).astype(float), levels=[0.5, 1.5],
                    colors=['0.85'])
    A0.set_title(f'{name}\n$T = {cyc.T:.1f}$ t.u.', fontsize=11)
    A1 = ax[1, k]
    im = A1.pcolormesh(G1, G2, np.log10(1 + mag), cmap='magma', shading='auto')
    A1.plot(*np.vstack([cyc.Y, cyc.Y[:1]]).T, 'w', lw=1.4)
    fig.colorbar(im, ax=A1, fraction=0.046, pad=0.03)
    for A in (A0, A1):
        A.set_aspect('equal'); A.set_xticks([]); A.set_yticks([])
        A.set_xlim(G1.min(), G1.max()); A.set_ylim(G2.min(), G2.max())
ax[0, 0].set_ylabel('isochron foliation')
ax[1, 0].set_ylabel(r'$\log_{10}(1+|\nabla\phi|)$')
plt.tight_layout()
plt.savefig('figures/fig1_foliations.png', dpi=170)
