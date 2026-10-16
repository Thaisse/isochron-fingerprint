"""In-silico versions of experimentally measurable quantities (IEEE item 4)
for every sweep point -> out/physio.json

Protocol (mirrors a dynamic-clamp PRC experiment): during tonic firing a
brief current pulse is modelled as an instantaneous voltage kick
dv = eps * (peak-to-peak V on Gamma), delivered at 40 phases.
Measured: first- and second-order PRCs from spike times (upward crossings
of the mid-amplitude voltage level), asymptotic phase shift, and whether
firing is annihilated.
"""
import json, numpy as np, time
from multiprocessing import Pool
from scipy.spatial import cKDTree
from isofp import core as c, sweep

EPS = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
NPH = 40
TWOPI = 2 * np.pi

def project_phase(cyc, Y, j):
    """Sub-sample phase: orthogonal projection of Y onto the two cycle
    segments adjacent to the nearest sample j (removes the 2*pi/M
    quantisation, needed for 1% pulses)."""
    M = len(cyc.Y); best_d = np.full(len(Y), np.inf); best_th = cyc.theta[j].copy()
    for a in (-1, 0):
        i0 = (j + a) % M; i1 = (i0 + 1) % M
        P0, P1 = cyc.Y[i0], cyc.Y[i1]
        d = P1 - P0
        s = np.clip(((Y - P0) * d).sum(1) / (d * d).sum(1), 0, 1)
        dist = np.hypot(*(P0 + s[:, None] * d - Y).T)
        th = cyc.theta[i0] + s * TWOPI / M
        upd = dist < best_d
        best_d[upd] = dist[upd]; best_th[upd] = th[upd]
    return best_th

def measure(f, p, x0, tt, n_per=8, spp=2000):
    cyc = c.limit_cycle(f, p, x0, tt)
    T, X = cyc.T, cyc.X
    M = len(X)
    vpp = np.ptp(X[:, 0]); vth = 0.5 * (X[:, 0].max() + X[:, 0].min())
    th_k = TWOPI * np.arange(NPH) / NPH
    idx = (np.arange(NPH) * M) // NPH
    amps = np.concatenate([EPS, -EPS])
    P, A = np.meshgrid(idx, amps, indexing='ij')
    v = X[P.ravel(), 0] + A.ravel() * vpp
    w = X[P.ravel(), 1].astype(float).copy()
    th0 = th_k[np.repeat(np.arange(NPH), len(amps))]
    n = n_per * spp; dt = n_per * T / n
    nT = len(v)
    spikes = np.full((nT, 3), np.nan); cnt = np.zeros(nT, int)
    t = 0.0
    with np.errstate(all='ignore'):
        for k in range(n):
            k1 = f(v, w, p); k2 = f(v + .5 * dt * k1[0], w + .5 * dt * k1[1], p)
            k3 = f(v + .5 * dt * k2[0], w + .5 * dt * k2[1], p)
            k4 = f(v + dt * k3[0], w + dt * k3[1], p)
            vn = v + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            wn = w + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            up = (v < vth) & (vn >= vth) & (cnt < 3)
            if up.any():
                fr = (vth - v[up]) / (vn[up] - v[up])
                spikes[up, cnt[up]] = t + fr * dt
                cnt[up] += 1
            v, w = vn, wn; t += dt
    # asymptotic phase from final state (time-uniform phase, nondim metric)
    Yf = np.stack([v * cyc.S[0], w * cyc.S[1]], 1)
    dist, j = cKDTree(cyc.Y).query(np.nan_to_num(Yf, nan=1e6))
    alive = dist <= 0.10 * cyc.D
    ph_end = np.mod(project_phase(cyc, Yf, j) - TWOPI * t / T, TWOPI)
    dphi = np.angle(np.exp(1j * (ph_end - th0)))          # asymptotic shift
    t_next = T * (1 - th0 / TWOPI)                       # unperturbed next spike
    F1 = (t_next - spikes[:, 0]) / T
    F2 = (T - (spikes[:, 1] - spikes[:, 0])) / T
    sh = (NPH, len(amps))
    return cyc, dict(dphi=dphi.reshape(sh), F1=F1.reshape(sh), F2=F2.reshape(sh),
                     alive=alive.reshape(sh))

def gini(x):
    x = np.sort(np.abs(x)); n = len(x); k = np.arange(1, n + 1)
    return 2 * (k * x).sum() / (n * x.sum()) - (n + 1) / n

def observables(r):
    ne = len(EPS)
    dp = r['dphi'][:, :ne]                         # positive kicks
    base = dp[:, 0] / EPS[0]                       # ~ Z_v * vpp
    o = {}
    o['prc_adv_frac'] = float((base > 0).mean())
    o['prc_kDC'] = float(abs(base.mean()) / np.sqrt((base ** 2).mean()))
    cc = np.fft.rfft(base); o['prc_k2'] = float(abs(cc[2]) / abs(cc[1]))
    o['prc_gini'] = float(gini(base))
    o['prc_logR'] = float(np.log10(np.percentile(np.abs(base), 98) /
                                   max(np.percentile(np.abs(base), 2), 1e-12)))
    # linear range: largest eps whose normalised PRC stays within 10%
    lin = EPS[0]
    for k, e in enumerate(EPS):
        ok = r['alive'][:, k] & r['alive'][:, ne + k]
        num = np.linalg.norm((dp[ok, k] / e - base[ok]))
        if ok.all() and num / np.linalg.norm(base) < 0.10:
            lin = e
        else:
            break
    o['eps_lin'] = float(lin)
    k = int(np.where(EPS == 0.05)[0][0])
    F1, F2 = r['F1'][:, k], r['F2'][:, k]
    good = np.isfinite(F1) & np.isfinite(F2)
    o['F2_over_F1'] = float(np.sqrt((F2[good] ** 2).mean()) / np.sqrt((F1[good] ** 2).mean()))
    o['annihilation_frac'] = float(1 - r['alive'].mean())
    return o

def job(k):
    fam, f, p, x0, tt = sweep.sweep_points()[k]
    t = time.time()
    cyc, r = measure(f, p, x0, tt)
    o = observables(r)
    print(k, fam, '%.1fs' % (time.time() - t), {a: round(b, 3) for a, b in o.items()}, flush=True)
    return dict(family=fam, params=p, **o)

if __name__ == '__main__':
    n = len(sweep.sweep_points())
    with Pool(2) as pool:
        res = pool.map(job, range(n), chunksize=1)
    json.dump(res, open('out/physio.json', 'w'), indent=1)
