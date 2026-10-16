"""Statistics of Section V: eta^2, AUC, extrapolation-split classification."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

FAMS = ['FHN', 'ML-II', 'ML-I', 'HR', 'SL']
METRIC = ['D1', 'CV', 'G', 'rc']
INVAR = ['nLT', 'kDC', 'k2', 'logR']
ALL = METRIC + INVAR
PRIMARY = {'FHN': 'I', 'ML-II': None, 'ML-I': 'I', 'HR': 'I', 'SL': 'beta'}


def table(res, lab):
    fam = np.array([r['family'] for r in res])
    X = {k: np.array([r[lab][k] for r in res]) for k in res[0][lab]}
    return fam, X


def eta2(x, fam):
    ok = np.isfinite(x); x, fam = x[ok], fam[ok]
    gm = x.mean()
    ssb = sum(((x[fam == f].mean() - gm) ** 2) * (fam == f).sum()
              for f in np.unique(fam))
    return ssb / ((x - gm) ** 2).sum()


def split_masks(res):
    """Low/high half of each family's parameter range (primary parameter,
    normalised; ML-II uses the mean normalised rank of I and gCa)."""
    fam = np.array([r['family'] for r in res])
    score = np.zeros(len(res))
    for f in np.unique(fam):
        idx = np.where(fam == f)[0]
        if f == 'ML-II':
            a = np.array([res[i]['params']['I'] for i in idx])
            b = np.array([res[i]['params']['gCa'] for i in idx])
            s = (a - a.min()) / np.ptp(a) + (b - b.min()) / np.ptp(b)
        else:
            k = PRIMARY[f]
            s = np.array([res[i]['params'][k] for i in idx], float)
        r = np.argsort(np.argsort(s + 1e-9 * np.arange(len(s))))
        score[idx] = r / (len(idx) - 1)
    return score < 0.5, score >= 0.5


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))


def extrap_acc(Xm, y, lo, hi):
    acc = []
    for tr, te in ((lo, hi), (hi, lo)):
        m = clf().fit(Xm[tr], y[tr])
        acc.append((m.predict(Xm[te]) == y[te]).mean())
    return float(np.mean(acc))


def classify(res, lab, feats, n_perm=300, seed=0):
    fam, X = table(res, lab)
    Xm = np.column_stack([X[k] for k in feats])
    lo, hi = split_masks(res)
    ok = np.isfinite(Xm).all(1)            # drop undefined descriptors
    Xm, fam, lo, hi = Xm[ok], fam[ok], lo[ok], hi[ok]
    a = extrap_acc(Xm, fam, lo, hi)
    rng = np.random.default_rng(seed)
    null = np.array([extrap_acc(Xm, rng.permutation(fam), lo, hi)
                     for _ in range(n_perm)])
    p = (1 + (null >= a).sum()) / (n_perm + 1)
    chance = max((fam == f).mean() for f in np.unique(fam))
    return a, p, chance


def auc(x, fam, a='FHN', b='ML-II'):
    m = ((fam == a) | (fam == b)) & np.isfinite(x)
    s = roc_auc_score((fam[m] == a).astype(int), x[m])
    return max(s, 1 - s)
