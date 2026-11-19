import os, sys, glob, time
import numpy as np
import pandas as pd
import cv2


def fid_path(out, obs):
    """fiducials of step 1: liver_out/step1/ (new layout) or liver_out/ (old layout)"""
    for p in (os.path.join(out, "step1", f"fiducials_observer{obs}.csv"),
              os.path.join(out, f"fiducials_observer{obs}.csv")):
        if os.path.exists(p):
            return p
    sys.exit(f"fiducials_observer{obs}.csv not found in {out} - run step 1 first, "
             f"and run this script from the folder that contains liver_out")

OUT = sys.argv[1] if len(sys.argv) > 1 else "liver_out"
FACTOR = int(sys.argv[2]) if len(sys.argv) > 2 else 8
UM_PER_PX = 0.46 * FACTOR
COARSE = 4                                   # extra downsampling for the angle search
MIRROR = "--mirror" in sys.argv               # also try mirrored sections (off: none found)
TIE = 0.03                                    # scores this close count as a tie
IMG_DIR = os.path.join(OUT, f"liver_ds{FACTOR}")
RES = os.path.join(OUT, "rigid")
os.makedirs(RES, exist_ok=True)

files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
names = [os.path.splitext(os.path.basename(f))[0] for f in files]
n = len(files)
print(f"{n} downsampled sections, {UM_PER_PX:.2f} um/px")


# ------------------------------------------------------------ fiducials, (x, y) in ds px
def load_fid(obs):
    df = pd.read_csv(fid_path(OUT, obs), dtype={"name": str})
    df["name"] = df["name"].str.zfill(3)
    df = df.set_index("name").loc[names]
    return np.stack([df[[f"x{i}_ds", f"y{i}_ds"]].to_numpy() for i in range(1, 5)], axis=1)

P1, P2 = load_fid(1), load_fid(2)


# ------------------------------------------------------------ transform helpers (3x3)
def to3(m):
    return np.vstack([m, [0, 0, 1]])


def apply(T, pts):
    p = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)
    return (p @ T.T)[..., :2]


def kabsch(src, dst, allow_reflection=False):
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.eye(2) if allow_reflection else np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return to3(np.hstack([R, (cd - R @ cs)[:, None]]))


def chain(pairwise):
    G = [np.eye(3)]
    for T in pairwise[1:]:
        G.append(G[-1] @ T)
    return G


# ------------------------------------------------------------ images
def feature_image(f):
    """inverted, smoothed grey: tissue bright, background and vessels dark"""
    g = 255 - cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 2.0).astype(np.float32)
    g[g < 15] = 0                                        # clean white background
    return g

print("loading ...")
imgs = [feature_image(f) for f in files]
h, w = imgs[0].shape
c8 = np.array([w / 2.0, h / 2.0])

# coarse images on a square canvas large enough for any rotation
hc, wc = h // COARSE, w // COARSE
S = int(2 ** np.ceil(np.log2(np.hypot(hc, wc) * 1.15)))
off = np.array([(S - wc) // 2, (S - hc) // 2])
cc = off + np.array([wc / 2.0, hc / 2.0])                # canvas point = image centre


def coarse_canvas(g):
    small = cv2.resize(g, (wc, hc), interpolation=cv2.INTER_AREA)
    can = np.zeros((S, S), np.float32)
    can[off[1]:off[1] + hc, off[0]:off[0] + wc] = small
    return can

coarse = [coarse_canvas(g) for g in imgs]


def best_shift(Ff, mov):
    """FFT cross-correlation; returns (score, dx, dy) with fixed(p) ~ mov(p - d)"""
    Fm = np.fft.rfft2(mov)
    c = np.fft.irfft2(Ff * np.conj(Fm), s=mov.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    dy = iy - S if iy > S // 2 else iy
    dx = ix - S if ix > S // 2 else ix
    return c[iy, ix], dx, dy


def flip_matrix(cx):
    """mirror x about the vertical line x = cx (3x3)"""
    return np.array([[-1.0, 0, 2 * cx], [0, 1, 0], [0, 0, 1]])


def global_search(fix, mov):
    """Rotation sweep (+ the mirrored section, in case it was placed upside
    down on the slide). Returns candidates sorted by score:
    (score, T_ds, angle, flipped)."""
    Ff = np.fft.rfft2(fix)
    nf = np.linalg.norm(fix)
    cands = []
    for flipped in ((False, True) if MIRROR else (False,)):
        m0 = cv2.warpAffine(mov, flip_matrix(cc[0])[:2], (S, S)) if flipped else mov
        res = {}

        def try_angle(a):
            Rm = cv2.getRotationMatrix2D(tuple(cc), a, 1.0)
            m = cv2.warpAffine(m0, Rm, (S, S), flags=cv2.INTER_LINEAR)
            v, dx, dy = best_shift(Ff, m)
            res[a] = (v / (nf * np.linalg.norm(m) + 1e-9), dx, dy)

        for a in np.arange(-180, 180, 3.0):
            try_angle(a)
        # refine the best angle and the best angle at least 30 deg away from it
        order = sorted(res, key=lambda a: -res[a][0])
        seeds = [order[0]]
        seeds += [a for a in order if abs((a - order[0] + 180) % 360 - 180) >= 30][:1]
        seeds += [a for a in order if abs(a) <= 45][:1]          # best small rotation
        seeds = list(dict.fromkeys(seeds))
        for a0 in seeds:
            sub = {}
            for a in np.arange(a0 - 3, a0 + 3.01, 0.25):
                try_angle(a); sub[a] = res[a]
            a = max(sub, key=lambda x: sub[x][0])
            v, dx, dy = sub[a]
            M = to3(cv2.getRotationMatrix2D(tuple(c8), a, 1.0))
            M[:2, 2] += COARSE * np.array([dx, dy])
            T = M @ flip_matrix(c8[0]) if flipped else M
            cands.append((v, T, a, flipped))
    return sorted(cands, key=lambda c: -c[0])


def ecc_refine(fix, mov, T0):
    """Pre-warp the moving image with T0, then let ECC find the small residual
    from identity (OpenCV's Euclidean ECC returns NaN when started from a
    large rotation). ECC warp maps fixed -> warped-moving coordinates, so the
    residual moving->fixed transform is its inverse; total = residual @ T0."""
    mov_w = cv2.warpAffine(mov, T0[:2], (fix.shape[1], fix.shape[0]), flags=cv2.INTER_LINEAR)
    W = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc_val, W = cv2.findTransformECC(fix, mov_w, W, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        return np.linalg.inv(to3(W.astype(np.float64))) @ T0, cc_val, True
    except cv2.error:
        return T0, np.nan, False


# ------------------------------------------------------------ run
def rms_um(T, src, dst):
    return np.sqrt(np.mean(np.sum((apply(T, src) - dst) ** 2, axis=1))) * UM_PER_PX


def landmark_pair(k):
    """rigid fit to the holes; a mirror is accepted only if it halves the residual"""
    Tn, Tm = kabsch(P1[k], P1[k - 1]), kabsch(P1[k], P1[k - 1], allow_reflection=True)
    return Tm if rms_um(Tm, P1[k], P1[k - 1]) < 0.5 * rms_um(Tn, P1[k], P1[k - 1]) else Tn

pair_lm = [np.eye(3)] + [landmark_pair(k) for k in range(1, n)]

pair_glob, pair_ecc, log = [np.eye(3)], [np.eye(3)], []
t0 = time.time()
for k in range(1, n):
    # landmark diagnostics: rotation of the section and whether a mirror fits better
    Tl, Tr = kabsch(P1[k], P1[k - 1]), kabsch(P1[k], P1[k - 1], allow_reflection=True)
    lm_ang = np.degrees(np.arctan2(Tl[1, 0], Tl[0, 0]))
    lm_rms, lm_rms_mirror = rms_um(Tl, P1[k], P1[k - 1]), rms_um(Tr, P1[k], P1[k - 1])

    cands = global_search(coarse[k - 1], coarse[k])
    # Tie rule: consecutive sections come from the same block, so among candidates
    # whose scores are within TIE of the best, take the smallest rotation (and an
    # unmirrored one). v3 let ECC break ties and it picked ~150-180 deg twice.
    ties = [c for c in cands if c[0] >= cands[0][0] - TIE]
    score, Tg, ang, fl = min(ties, key=lambda c: (c[3], abs(c[2])))
    Te, ecc_cc, ok = ecc_refine(imgs[k - 1], imgs[k], Tg)
    to_try = ties
    ang = np.degrees(np.arctan2(Te[1, 0], Te[0, 0]))     # same convention as landmarks
    pair_glob.append(Tg); pair_ecc.append(Te)
    log.append(dict(pair=f"{names[k]}->{names[k-1]}", global_score=score,
                    runner_up_score=cands[1][0], candidates_tried=len(to_try),
                    flipped=fl, angle_global=ang, ecc_cc=ecc_cc, ecc_ok=ok,
                    landmark_angle=lm_ang, landmark_rms_um=lm_rms,
                    landmark_rms_if_mirrored_um=lm_rms_mirror))
    print(f"  {names[k]} -> {names[k-1]}: {'MIRRORED ' if fl else ''}rotation {ang:+7.2f} deg"
          f" (score {score:.3f}, best {cands[0][0]:.3f}, ties {len(to_try)})"
          f"  ECC {'ok' if ok else 'FAILED'} cc={ecc_cc:.3f}"
          f" | landmarks: {lm_ang:+7.2f} deg, rms {lm_rms:6.1f} um"
          f" (mirrored {lm_rms_mirror:6.1f})")
print(f"registration took {time.time() - t0:.0f} s")
pd.DataFrame(log).to_csv(os.path.join(RES, "pair_log.csv"), index=False)
G = {"unregistered": [np.eye(3)] * n,
     "rigid, landmarks (reference)": chain(pair_lm),
     "rigid, global search": chain(pair_glob),
     "rigid, global search + ECC": chain(pair_ecc)}


# ------------------------------------------------------------ evaluation
def tre(Gk, P):
    Q = np.stack([apply(Gk[k], P[k]) for k in range(n)])
    return np.linalg.norm(Q[1:] - Q[:-1], axis=2) * UM_PER_PX

rows, per_pair = [], {}
for label, Gk in G.items():
    for obs, P in [(1, P1), (2, P2)]:
        d = tre(Gk, P)
        rows.append(dict(method=label, observer=obs, mean_um=d.mean(), sd_um=d.std(ddof=1),
                         median_um=np.median(d), p95_um=np.percentile(d, 95), n=d.size))
        if obs == 1:
            per_pair[label] = d.mean(axis=1)
io = np.linalg.norm(P1 - P2, axis=2) * UM_PER_PX
rows.append(dict(method="inter-observer (lower bound)", observer=None, mean_um=io.mean(),
                 sd_um=io.std(ddof=1), median_um=np.median(io),
                 p95_um=np.percentile(io, 95), n=io.size))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RES, "tre_summary.csv"), index=False)
pd.DataFrame(per_pair, index=[f"{a}-{b}" for a, b in zip(names[:-1], names[1:])]
             ).to_csv(os.path.join(RES, "tre_per_pair.csv"))
print("\nTRE between consecutive sections [um]  (n = hole pairs)")
print(summ.round(1).to_string(index=False))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8.5, 4.2))
for label, v in per_pair.items():
    ax.semilogy(range(1, len(v) + 1), v, "o-", ms=4, label=label)
ax.axhline(np.median(io), color="k", ls=":", label="inter-observer (median)")
ax.set_xlabel("section pair k, k+1"); ax.set_ylabel("mean TRE of 4 holes [µm]")
ax.set_title(f"Rigid registration of {n} liver sections ({UM_PER_PX:.2f} µm/px), observer 1")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(RES, "tre_per_pair.png"), dpi=150)


# ------------------------------------------------------------ registered stack + reslice
best = G["rigid, global search + ECC"]
pad = int(0.3 * max(h, w))
Tpad = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], float)
H, W = h + 2 * pad, w + 2 * pad
dst_dir = os.path.join(RES, "registered"); os.makedirs(dst_dir, exist_ok=True)
y_cut = int(np.mean(P1[0][:, 1])) + pad                  # horizontal line through the holes
rb, ra = [], []
for k, f in enumerate(files):
    img = cv2.imread(f)
    reg = cv2.warpAffine(img, (Tpad @ best[k])[:2], (W, H), borderValue=(255, 255, 255))
    raw = cv2.warpAffine(img, Tpad[:2], (W, H), borderValue=(255, 255, 255))
    cv2.imwrite(os.path.join(dst_dir, names[k] + ".png"), reg)
    rb.append(raw[y_cut]); ra.append(reg[y_cut])
Z = 8          # each 5 um section drawn 8 rows high so the cut is visible (not to scale)
rb = np.repeat(np.stack(rb), Z, axis=0); ra = np.repeat(np.stack(ra), Z, axis=0)
cols = np.where((np.vstack([rb, ra]).min(axis=(0, 2)) < 240))[0]
x0, x1 = (cols.min(), cols.max() + 1) if cols.size else (0, W)
gap = np.full((20, x1 - x0, 3), 255, np.uint8)
cv2.imwrite(os.path.join(RES, "reslice_before_after.png"),
            np.vstack([rb[:, x0:x1], gap, ra[:, x0:x1]]))
print("registered stack and figures written to", RES)
