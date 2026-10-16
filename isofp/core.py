"""Core of the isochron-foliation fingerprint framework.

Pipeline (Section III of the paper):
  1. limit cycle extraction            -> limit_cycle()
  2. non-dimensionalisation             -> Cycle.S
  3. grid specification                 -> make_grid()
  4. vectorised forward integration     -> integrate_grid()
  5. phase assignment                   -> assign_phase()
  6. wrap-aware gradient                -> phase_gradient()
  7. descriptors D1-D6                  -> descriptors()
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator

TWOPI = 2 * np.pi


class Cycle:
    """Limit cycle sampled uniformly in time (asymptotic phase) and the
    diagonal scaling S that maps it to a unit bounding box."""

    def __init__(self, t, X, T, f, p, M):
        self.T, self.f, self.p = T, f, p
        self.X = X                                   # native coords, M x 2
        ext = X.max(0) - X.min(0)
        self.S = 1.0 / ext                           # y = S * x
        self.Y = X * self.S                          # nondim coords
        self.theta = TWOPI * np.arange(M) / M        # time-uniform phase
        # arc-length labelling in native coordinates (used in v10; NOT the
        # asymptotic phase -- kept only to quantify its effect)
        seg = np.hypot(*np.diff(np.vstack([X, X[:1]]), axis=0).T)
        s_ = np.r_[0, np.cumsum(seg)]
        self.theta_arc = TWOPI * s_[:-1] / s_[-1]
        self.centroid = self.Y.mean(0)
        d = self.Y[:, None, :] - self.Y[None, ::4, :]
        self.D = float(np.sqrt((d ** 2).sum(-1)).max())   # Feret diameter

    def g(self, y1, y2):
        """Vector field in nondimensional coordinates g = S f(S^-1 y)."""
        a, b = self.f(y1 / self.S[0], y2 / self.S[1], self.p)
        return a * self.S[0], b * self.S[1]


def limit_cycle(f, p, x0, t_trans, M=1200, rtol=1e-10, atol=1e-12,
                section=None):
    """Integrate off the transient, locate the period with a Poincare
    section (upward crossing of x = section) and resample one period
    uniformly in time at M points."""
    rhs = lambda t, z: np.array(f(z[0], z[1], p))
    s = solve_ivp(rhs, (0, t_trans), x0, method='DOP853', rtol=rtol,
                  atol=atol)
    z = s.y[:, -1]
    s = solve_ivp(rhs, (0, t_trans), z, method='DOP853', rtol=rtol,
                  atol=atol, dense_output=True, max_step=t_trans / 2000)
    xs = s.y[0]
    if section is None:
        section = 0.5 * (xs.min() + xs.max())
    idx = np.where((xs[:-1] < section) & (xs[1:] >= section))[0]
    if len(idx) < 3 or (xs.max() - xs.min()) < 1e-6:
        raise RuntimeError('no limit cycle found')
    def cross(i):
        a, b = s.t[i], s.t[i + 1]
        for _ in range(60):               # bisection on dense output
            m = 0.5 * (a + b)
            if s.sol(m)[0] < section:
                a = m
            else:
                b = m
        return 0.5 * (a + b)
    t0, t1 = cross(idx[-3]), cross(idx[-2])
    T = t1 - t0
    t = t0 + T * np.arange(M) / M
    X = s.sol(t).T
    return Cycle(t - t0, X, T, f, p, M)


def make_grid(cyc, N, half_width=0.9, offset=(0.0, 0.0)):
    """Square grid of half-width half_width*D centred on the cycle
    centroid (nondimensional coordinates).  `offset` shifts the grid by a
    fraction of the grid spacing (used for the stability analysis)."""
    hw = half_width * cyc.D
    h = 2 * hw / (N - 1)
    c = cyc.centroid + np.asarray(offset) * h
    g1 = c[0] - hw + h * np.arange(N)
    g2 = c[1] - hw + h * np.arange(N)
    Y1, Y2 = np.meshgrid(g1, g2, indexing='ij')
    return g1, g2, Y1, Y2, h


def integrate_grid(cyc, Y1, Y2, n_periods=4.0, dt=None, method='rk4',
                   dtype=np.float64):
    """Integrate all grid points simultaneously for n_periods * T."""
    T = cyc.T
    if dt is None:
        dt = min(0.02, T / 400)
    n = int(np.ceil(n_periods * T / dt))
    dt = n_periods * T / n              # land exactly on n_periods*T
    a, b = Y1.astype(dtype).copy(), Y2.astype(dtype).copy()
    g = cyc.g
    with np.errstate(all='ignore'):
        for _ in range(n):
            if method == 'euler':
                k1 = g(a, b)
                a = a + dt * k1[0]; b = b + dt * k1[1]
            elif method == 'rk4':
                k1 = g(a, b)
                k2 = g(a + 0.5 * dt * k1[0], b + 0.5 * dt * k1[1])
                k3 = g(a + 0.5 * dt * k2[0], b + 0.5 * dt * k2[1])
                k4 = g(a + dt * k3[0], b + dt * k3[1])
                a = a + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
                b = b + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            else:
                raise ValueError(method)
            # freeze diverging points (they are phaseless anyway)
            bad = ~np.isfinite(a) | ~np.isfinite(b) | (np.abs(a) > 1e3) | (np.abs(b) > 1e3)
            if bad.any():
                a[bad] = 1e3; b[bad] = 1e3
    return a.astype(np.float64), b.astype(np.float64), n * dt


def assign_phase(cyc, A, B, t_elapsed, dmax=0.10, labelling='time'):
    """Phase = phase of nearest cycle point to the final state minus the
    phase advance 2*pi*t/T.  Points farther than dmax*D are phaseless."""
    tree = cKDTree(cyc.Y)
    pts = np.stack([A.ravel(), B.ravel()], 1)
    d, i = tree.query(pts)
    th = np.mod(cyc.theta[i] - TWOPI * t_elapsed / cyc.T, TWOPI)
    if labelling == 'arc':          # monotone relabelling theta -> s(theta)
        th = np.interp(th, np.r_[cyc.theta, TWOPI],
                       np.r_[cyc.theta_arc, TWOPI])
    th = th.reshape(A.shape)
    ok = (d.reshape(A.shape) <= dmax * cyc.D)
    th[~ok] = np.nan
    return th


def phase_gradient(th, h, wrap_aware=True):
    """Second-order central differences.  wrap_aware: difference e^{i phi}
    and use d phi = Im(e^{-i phi} d e^{i phi})."""
    if wrap_aware:
        E = np.exp(1j * th)
        d1, d2 = np.gradient(E, h, h)
        Em = np.conj(E)
        g1, g2 = np.imag(Em * d1), np.imag(Em * d2)
    else:
        g1, g2 = np.gradient(th, h, h)
    return g1, g2


def floquet_contraction(cyc, eps=1e-6):
    """-Lambda*T = -int_0^T tr J(Gamma(t)) dt (coordinate invariant)."""
    x, y = cyc.X[:, 0], cyc.X[:, 1]
    f = cyc.f
    ex = eps * max(1.0, np.abs(x).max()); ey = eps * max(1.0, np.abs(y).max())
    dfx = (f(x + ex, y, cyc.p)[0] - f(x - ex, y, cyc.p)[0]) / (2 * ex)
    dgy = (f(x, y + ey, cyc.p)[1] - f(x, y - ey, cyc.p)[1]) / (2 * ey)
    tr = dfx + dgy
    return -tr.mean() * cyc.T          # periodic trapezoid on uniform grid


def adjoint_prc(cyc, n_back=20, eps=1e-6):
    """Reference iPRC Z(theta) in NATIVE coordinates by the adjoint method
    (backward integration of dZ/dt = -J^T Z along Gamma, normalised by
    Z . f = 2*pi/T).  Used only as ground truth for the grid-based PRC."""
    from scipy.interpolate import CubicSpline
    M = len(cyc.X)
    tt = cyc.T * np.arange(M + 1) / M
    Xp = np.vstack([cyc.X, cyc.X[:1]])
    sx = CubicSpline(tt, Xp, bc_type='periodic')
    f, p = cyc.f, cyc.p
    def J(t):
        x, y = sx(np.mod(t, cyc.T))
        ex = eps * max(1, abs(x)); ey = eps * max(1, abs(y))
        a = (np.array(f(x + ex, y, p)) - np.array(f(x - ex, y, p))) / (2 * ex)
        b = (np.array(f(x, y + ey, p)) - np.array(f(x, y - ey, p))) / (2 * ey)
        return np.array([[a[0], b[0]], [a[1], b[1]]])
    rhs = lambda t, z: -J(t).T @ z           # integrated backward in time
    z = np.array([1.0, 0.0])
    s = solve_ivp(lambda t, z: rhs(t, z), (n_back * cyc.T, 0), z,
                  method='DOP853', rtol=1e-9, atol=1e-12,
                  t_eval=np.linspace(cyc.T, 0, M + 1))
    Z = s.y[:, ::-1][:, :M].T                 # times 0..T(1-1/M)
    F = np.array(f(cyc.X[:, 0], cyc.X[:, 1], p)).T
    norm = (Z * F).sum(1)
    Z = Z * (TWOPI / cyc.T) / norm[:, None]
    return Z


def gini(v):
    v = np.sort(v)
    n = len(v)
    k = np.arange(1, n + 1)
    return 2 * (k * v).sum() / (n * v.sum()) - (n + 1) / n


def descriptors(cyc, g1, g2, G1, G2, th, gr1, gr2, alpha=2.0,
                volt_axis=0, Z_ref=None):
    """Compute D1-D6 and log10 R_Gamma in nondimensional coordinates.
    Quantities on Gamma (threshold of D4, D6, R_Gamma) use the adjoint iPRC
    Z_ref (nondimensional) when given; the grid-interpolated iPRC is kept
    as a consistency check (keys *_grid)."""
    mag = np.hypot(gr1, gr2)
    ok = np.isfinite(mag)
    m = mag[ok]
    out = {}
    out['D1'] = m.mean() * cyc.D
    out['CV'] = m.std() / m.mean()
    out['G'] = gini(m)
    # gradient on Gamma by bilinear interpolation
    fill = lambda a: np.where(np.isfinite(a), a, 0.0)
    I1 = RegularGridInterpolator((g1, g2), fill(gr1), bounds_error=False)
    I2 = RegularGridInterpolator((g1, g2), fill(gr2), bounds_error=False)
    Zg = np.stack([I1(cyc.Y), I2(cyc.Y)], 1)       # nondim iPRC
    Zuse = Zg if Z_ref is None else Z_ref
    magG = np.hypot(Zuse[:, 0], Zuse[:, 1])
    tree = cKDTree(cyc.Y)
    dist, _ = tree.query(np.stack([G1[ok], G2[ok]], 1))
    thr = alpha * magG.mean()
    sel = m > thr
    out['rc'] = dist[sel].mean() / cyc.D if sel.any() else np.nan
    out['nLT'] = floquet_contraction(cyc)
    def kap(Z):
        Zv = Z[:, volt_axis]
        cc = np.fft.rfft(Zv) / len(Zv)
        mg = np.hypot(Z[:, 0], Z[:, 1])
        return (abs(Zv.mean()) / np.sqrt((Zv ** 2).mean()),
                abs(cc[2]) / abs(cc[1]),
                np.log10(np.percentile(mg, 98) / np.percentile(mg, 2)))
    out['kDC'], out['k2'], out['logR'] = kap(Zuse)
    out['kDC_grid'], out['k2_grid'], out['logR_grid'] = kap(Zg)
    if Z_ref is not None:
        out['prc_relerr'] = (np.linalg.norm(Zg[:, volt_axis] - Z_ref[:, volt_axis])
                             / np.linalg.norm(Z_ref[:, volt_axis]))
    out['phaseless_frac'] = 1 - ok.mean()
    out['_Zg'] = Zg
    return out


def run(f, p, x0, t_trans, N=72, half_width=0.9, n_periods=4.0, dt=None,
        method='rk4', dmax=0.10, alpha=2.0, M=1200, offset=(0, 0),
        wrap_aware=True, cyc=None, dtype=np.float64, return_all=False,
        labelling=('time',), use_adjoint=True, Zr=None):
    if cyc is None:
        cyc = limit_cycle(f, p, x0, t_trans, M=M)
    g1, g2, G1, G2, h = make_grid(cyc, N, half_width, offset)
    A, B, te = integrate_grid(cyc, G1, G2, n_periods, dt, method, dtype)
    if Zr is None and use_adjoint:
        Zr = adjoint_prc(cyc) / cyc.S
    res = {}
    for lab in labelling:
        th = assign_phase(cyc, A, B, te, dmax, lab)
        gr1, gr2 = phase_gradient(th, h, wrap_aware)
        d = descriptors(cyc, g1, g2, G1, G2, th, gr1, gr2, alpha,
                        Z_ref=Zr if lab == 'time' else None)
        d['T'] = cyc.T
        res[lab] = d
    if len(labelling) == 1:
        d = res[labelling[0]]
    else:
        d = res
    if return_all:
        return d, dict(cyc=cyc, g1=g1, g2=g2, G1=G1, G2=G2, th=th,
                       gr1=gr1, gr2=gr2, h=h)
    return d
