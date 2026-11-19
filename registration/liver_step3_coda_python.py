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
TILE_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
SRC_F = 8                                   # liver_ds8 = 3.68 um/px
WORK = 2                                    # extra factor -> 7.36 um/px (CODA: 8 um/px)
UM = 0.46 * SRC_F * WORK
COARSE = 4                                  # angle search at 29.4 um/px
SPACING_MM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5   # tile interval (paper: 1.5 mm)
SMOOTH_PX = 2.0                             # Gaussian sigma on the field (paper)
N_REF = 3                                   # candidate references (paper)
TIE = 0.03
MODE = sys.argv[4].lower() if len(sys.argv) > 4 else "paper"
if MODE not in ("paper", "refined"):
    sys.exit("MODE must be 'paper' or 'refined'")
UM_GLOBAL = 80.0                            # paper: global step at 80 um/px
RES = os.path.join(OUT, f"coda_python_{MODE}_tile{TILE_MM:g}_spacing{SPACING_MM:g}")
for d in ("", "registered_global", "registered_elastic"):
    os.makedirs(os.path.join(RES, d), exist_ok=True)

files = sorted(glob.glob(os.path.join(OUT, f"liver_ds{SRC_F}", "*.png")))
names = [os.path.splitext(os.path.basename(f))[0] for f in files]
n = len(files)
if n == 0:
    sys.exit(f"no images in {os.path.join(OUT, 'liver_ds8')} - run this script from the folder "
             f"that contains liver_out (e.g. D:\\master\\liver_pipeline), after step 1")
c_idx = n // 2                              # 0-based index of the reference section
print(f"{n} sections, mode {MODE}, working resolution {UM:.2f} um/px, reference {names[c_idx]}, "
      f"tiles {TILE_MM} mm every {SPACING_MM} mm")

# ------------------------------------------------------------ images on a padded frame
col0 = cv2.imread(files[0])
h0, w0 = col0.shape[0] // WORK, col0.shape[1] // WORK
pad = int(0.15 * max(h0, w0))
H, W = h0 + 2 * pad, w0 + 2 * pad


def load(f):
    c = cv2.resize(cv2.imread(f), (w0, h0), interpolation=cv2.INTER_AREA)
    c = cv2.copyMakeBorder(c, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    g = 255 - cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.5).astype(np.float32)
    g[g < 15] = 0
    return c, g

print("loading ...")
data = [load(f) for f in files]
colour = [d[0] for d in data]
grey = [d[1] for d in data]


def load_fid(obs):
    df = pd.read_csv(fid_path(OUT, obs), dtype={"name": str})
    df["name"] = df["name"].str.zfill(3)
    df = df.set_index("name").loc[names]
    P = np.stack([df[[f"x{i}_ds", f"y{i}_ds"]].to_numpy() for i in range(1, 5)], axis=1)
    return P / WORK + pad                    # (n, 4, 2) in padded working px

P1, P2 = load_fid(1), load_fid(2)


# ------------------------------------------------------------ helpers
def to3(m):
    return np.vstack([m, [0, 0, 1]])


def apply(T, pts):
    p = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)
    return (p @ T.T)[..., :2]


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return to3(np.hstack([R, (cd - R @ cs)[:, None]]))


def ncc(a, b, mask=None):
    m = (a > 0) & (b > 0) if mask is None else mask
    if m.sum() < 100:
        return -1.0
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


# ------------------------------------------------------------ global (rigid) step
hc, wc = H // COARSE, W // COARSE
S = int(np.ceil(np.hypot(hc, wc) * 1.1 / 32) * 32)
off = np.array([(S - wc) // 2, (S - hc) // 2])
cc = off + np.array([wc / 2.0, hc / 2.0])
cW = np.array([W / 2.0, H / 2.0])


def canvas(g):
    can = np.zeros((S, S), np.float32)
    can[off[1]:off[1] + hc, off[0]:off[0] + wc] = cv2.resize(g, (wc, hc), interpolation=cv2.INTER_AREA)
    return can


def best_shift(Ff, mov):
    c = np.fft.irfft2(Ff * np.conj(np.fft.rfft2(mov)), s=mov.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    return c[iy, ix], (ix - S if ix > S // 2 else ix), (iy - S if iy > S // 2 else iy)


def global_search(fix, mov):
    Ff, nf, res = np.fft.rfft2(fix), np.linalg.norm(fix), {}

    def try_angle(a):
        m = cv2.warpAffine(mov, cv2.getRotationMatrix2D(tuple(cc), a, 1.0), (S, S))
        v, dx, dy = best_shift(Ff, m)
        res[a] = (v / (nf * np.linalg.norm(m) + 1e-9), dx, dy)

    for a in np.arange(-180, 180, 3.0):
        try_angle(a)
    order = sorted(res, key=lambda a: -res[a][0])
    seeds = [order[0]]
    seeds += [a for a in order if abs((a - order[0] + 180) % 360 - 180) >= 30][:1]
    seeds += [a for a in order if abs(a) <= 45][:1]
    cands = []
    for a0 in dict.fromkeys(seeds):
        sub = {}
        for a in np.arange(a0 - 3, a0 + 3.01, 0.25):
            try_angle(a); sub[a] = res[a]
        a = max(sub, key=lambda x: sub[x][0])
        M = to3(cv2.getRotationMatrix2D(tuple(cW), a, 1.0))
        M[:2, 2] += COARSE * np.array(sub[a][1:])
        cands.append((sub[a][0], M, a))
    cands.sort(key=lambda c: -c[0])
    ties = [c for c in cands if c[0] >= cands[0][0] - TIE]
    return min(ties, key=lambda c: abs(c[2]))             # smallest rotation among ties


def ecc_refine(fix, mov, T0):
    mov_w = cv2.warpAffine(mov, T0[:2], (W, H))
    try:
        _, Wm = cv2.findTransformECC(fix, mov_w, np.eye(2, 3, dtype=np.float32),
                                     cv2.MOTION_EUCLIDEAN,
                                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6),
                                     None, 5)
        return np.linalg.inv(to3(Wm.astype(np.float64))) @ T0
    except cv2.error:
        return T0



# ------------------------------------------------------------ global step as in the paper
# 80 um/px canvas, large enough that rotations do not cut the tissue
G = UM_GLOBAL / UM                          # working px per 80-um px
hg, wg = int(round(H / G)), int(round(W / G))
Sg = int(np.ceil(np.hypot(hg, wg) * 1.05 / 2) * 2)
offg = np.array([(Sg - wg) // 2, (Sg - hg) // 2])
ccg = offg + np.array([wg / 2.0, hg / 2.0])  # canvas point that corresponds to the frame centre
RADON_ANG = np.arange(0, 180, 1.0)          # projection angles (deg)
NK = 40                                     # |FFT| coefficients kept per projection


def canvas80(g):
    can = np.zeros((Sg, Sg), np.float32)
    can[offg[1]:offg[1] + hg, offg[0]:offg[0] + wg] = cv2.resize(g, (wg, hg), interpolation=cv2.INTER_AREA)
    return can


def radon(img):
    """discrete Radon transform: one projection (column sums) per angle -> (n_angles, Sg)"""
    c = (Sg / 2.0, Sg / 2.0)
    return np.stack([cv2.warpAffine(img, cv2.getRotationMatrix2D(c, -t, 1.0), (Sg, Sg)).sum(0)
                     for t in RADON_ANG])


def radon_descriptor(img):
    """|FFT| of every projection: unchanged by a translation of the image, so only the
    rotation shifts it (along the angle axis). Normalised per frequency."""
    D = np.abs(np.fft.rfft(radon(img), axis=1))[:, 1:NK + 1]
    D = D - D.mean(0)
    return D / (np.linalg.norm(D, axis=0) + 1e-9)


def radon_angle(fix, mov):
    """rotation (deg, mod 180) that best aligns the Radon transforms of fix and mov"""
    Df, Dm = radon_descriptor(fix), radon_descriptor(mov)
    # circular cross-correlation along the angle axis, summed over frequencies
    score = np.fft.irfft(np.fft.rfft(Df, axis=0) * np.conj(np.fft.rfft(Dm, axis=0)),
                         n=len(RADON_ANG), axis=0).sum(1)
    return float(RADON_ANG[np.argmax(score)])


def best_shift_g(Ff, mov):
    c = np.fft.irfft2(Ff * np.conj(np.fft.rfft2(mov)), s=mov.shape)
    iy, ix = np.unravel_index(np.argmax(c), c.shape)
    return c[iy, ix], (ix - Sg if ix > Sg // 2 else ix), (iy - Sg if iy > Sg // 2 else iy)


def global_paper(fix, mov):
    """paper: angle from Radon cross-correlation, translation from image cross-correlation.
    The Radon angle is defined mod 180 deg; both candidates are tested."""
    cf, cm = canvas80(fix), canvas80(mov)
    t = radon_angle(cf, cm)
    Ff, nf, best = np.fft.rfft2(cf), np.linalg.norm(cf), None
    for a in (t, t - 180.0, -t, 180.0 - t):          # sign convention handled by testing both
        m = cv2.warpAffine(cm, cv2.getRotationMatrix2D(tuple(ccg), a, 1.0), (Sg, Sg))
        v, dx, dy = best_shift_g(Ff, m)
        v /= nf * np.linalg.norm(m) + 1e-9
        if best is None or v > best[0]:
            best = (v, a, dx, dy)
    v, a, dx, dy = best
    a = (a + 180.0) % 360.0 - 180.0
    M = to3(cv2.getRotationMatrix2D(tuple(cW), a, 1.0))
    M[:2, 2] += G * np.array([dx, dy])
    return v, M, a

# ------------------------------------------------------------ elastic step (tiles)
sp = SPACING_MM * 1000 / UM                 # tile interval in px
half = int(TILE_MM * 1000 / UM / 2)         # half tile size in px
nx, ny = int(np.ceil(W / sp)), int(np.ceil(H / sp))
gx = (np.arange(nx) + 0.5) * W / nx
gy = (np.arange(ny) + 0.5) * H / ny
win = np.outer(np.hanning(2 * half), np.hanning(2 * half)).astype(np.float32)


def tile_shift(F, M):
    """small rigid registration of one tile; returns backward displacement u at the
    tile centre (fixed(c) ~ moving(c + u)) and the tile correlation"""
    L = 2 * half
    cen = (half, half)
    Ff = np.fft.rfft2((F - F.mean()) * win)
    best = (-1, 0, 0, 0)
    for a in np.arange(-4, 4.01, 1.0):
        Rm = cv2.warpAffine(M, cv2.getRotationMatrix2D(cen, a, 1.0), (L, L))
        Mm = (Rm - Rm.mean()) * win
        c = np.fft.irfft2(Ff * np.conj(np.fft.rfft2(Mm)), s=(L, L))
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        dx = ix - L if ix > L // 2 else ix
        dy = iy - L if iy > L // 2 else iy
        v = c[iy, ix] / (np.linalg.norm((F - F.mean()) * win) * np.linalg.norm(Mm) + 1e-9)
        if v > best[0]:
            best = (v, a, dx, dy)
    v, a, dx, dy = best
    # fixed(p) ~ Rm(p - d), Rm(q) = M(R^-1 (q - c) + c)  =>  u = R^-1 (-d)
    R = cv2.getRotationMatrix2D((0, 0), a, 1.0)[:, :2]
    u = np.linalg.inv(R) @ (-np.array([dx, dy], float))
    return u, v


def elastic_field(fix, mov):
    """backward displacement field u (H, W, 2): registered(x) = mov(x + u(x))"""
    U = np.zeros((ny, nx, 2), np.float32)
    ok = np.zeros((ny, nx), np.float32)
    for j, y in enumerate(gy):
        for i, x in enumerate(gx):
            y0, x0 = int(y) - half, int(x) - half
            if y0 < 0 or x0 < 0 or y0 + 2 * half > H or x0 + 2 * half > W:
                continue
            F = fix[y0:y0 + 2 * half, x0:x0 + 2 * half]
            M = mov[y0:y0 + 2 * half, x0:x0 + 2 * half]
            if (F > 0).mean() < 0.5 or (M > 0).mean() < 0.5:
                continue                                    # mostly background
            u, v = tile_shift(F, M)
            if v < 0.2 or np.hypot(*u) > half / 2:
                continue                                    # unreliable tile
            U[j, i], ok[j, i] = u, 1
    if ok.sum() == 0:
        return np.zeros((H, W, 2), np.float32), 0
    # fill tiles without a result from their neighbours (normalised convolution)
    for _ in range(max(nx, ny)):
        if ok.min() > 0:
            break
        k = np.ones((3, 3), np.float32)
        num = np.stack([cv2.filter2D(U[..., c] * ok, -1, k, borderType=cv2.BORDER_CONSTANT)
                        for c in range(2)], -1)
        den = cv2.filter2D(ok, -1, k, borderType=cv2.BORDER_CONSTANT)
        fill = (ok == 0) & (den > 0)
        U[fill] = num[fill] / den[fill][:, None]
        ok[fill] = 1
    field = cv2.resize(U, (W, H), interpolation=cv2.INTER_LINEAR)
    field = np.stack([cv2.GaussianBlur(field[..., c], (0, 0), SMOOTH_PX) for c in range(2)], -1)
    return field, int(ok.sum())


xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))


def warp_elastic(img, field):
    return cv2.remap(img, xs + field[..., 0], ys + field[..., 1], cv2.INTER_LINEAR,
                     borderValue=(255, 255, 255) if img.ndim == 3 else 0)


def points_elastic(q, field):
    """forward-map points through the backward field: solve x + u(x) = q"""
    x = q.copy()
    for _ in range(10):
        ix = np.clip(np.round(x[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.round(x[:, 1]).astype(int), 0, H - 1)
        x = q - field[iy, ix]
    return x


# ------------------------------------------------------------ register outward from the centre
Tg = [None] * n                      # padded raw -> frame (global)
fields = [None] * n
reg_g = [None] * n                   # grey, globally registered
reg_e = [None] * n                   # grey, globally + elastically registered
Tg[c_idx], fields[c_idx] = np.eye(3), np.zeros((H, W, 2), np.float32)
reg_g[c_idx] = reg_e[c_idx] = grey[c_idx]
coarse_cache = {}
order = [k for m in range(1, n) for k in (c_idx - m, c_idx + m) if 0 <= k < n]
log, t0 = [], time.time()
for k in order:
    step = -1 if k < c_idx else 1                      # towards the centre
    refs = [k - step * m for m in range(1, N_REF + 1)
            if 0 <= k - step * m < n and reg_e[k - step * m] is not None]
    if MODE == "refined" and k not in coarse_cache:
        coarse_cache[k] = canvas(grey[k])
    best = None
    for r in refs:
        if MODE == "paper":
            score, T, a = global_paper(reg_e[r], grey[k])
        else:
            score, T, a = global_search(canvas(reg_e[r]), coarse_cache[k])
            T = ecc_refine(reg_e[r], grey[k], T)
        g_w = cv2.warpAffine(grey[k], T[:2], (W, H))
        q = ncc(reg_e[r], g_w)
        if best is None or q > best[0]:
            best = (q, r, T, g_w)
    q, r, T, g_w = best
    Tg[k], reg_g[k] = T, g_w
    field, ntiles = elastic_field(reg_e[r], g_w)
    fields[k] = field
    reg_e[k] = warp_elastic(g_w, field)
    q_e = ncc(reg_e[r], reg_e[k])
    ang = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    log.append(dict(section=names[k], reference=names[r], ncc_global=q, ncc_elastic=q_e,
                    angle=ang, tiles_used=ntiles,
                    max_disp_um=float(np.abs(field).max() * UM)))
    print(f"  {names[k]} -> ref {names[r]}  rot {ang:+7.2f} deg  NCC {q:.3f} -> {q_e:.3f}"
          f"  ({ntiles} tiles, max disp {np.abs(field).max() * UM:5.0f} um)"
          f"  [{time.time() - t0:5.0f} s]")
pd.DataFrame(log).to_csv(os.path.join(RES, "pair_log.csv"), index=False)


# ------------------------------------------------------------ evaluation
def tre_from_positions(Q):
    return np.linalg.norm(Q[1:] - Q[:-1], axis=2) * UM          # (n-1, 4)


def positions(P, elastic):
    Q = []
    for k in range(n):
        q = apply(Tg[k], P[k])
        Q.append(points_elastic(q, fields[k]) if elastic else q)
    return np.stack(Q)

pair_lm = [np.eye(3)] + [kabsch(P1[k], P1[k - 1]) for k in range(1, n)]
G_lm = [np.eye(3)]
for T in pair_lm[1:]:
    G_lm.append(G_lm[-1] @ T)
methods = {
    "unregistered": lambda P: P,
    "rigid, landmarks (reference)": lambda P: np.stack([apply(G_lm[k], P[k]) for k in range(n)]),
    "CODA-style global (rigid)": lambda P: positions(P, False),
    "CODA-style global + elastic": lambda P: positions(P, True),
}
rows, per_pair = [], {}
for label, fn in methods.items():
    for obs, P in [(1, P1), (2, P2)]:
        d = tre_from_positions(fn(P))
        rows.append(dict(method=label, observer=obs, mean_um=d.mean(), sd_um=d.std(ddof=1),
                         median_um=np.median(d), p95_um=np.percentile(d, 95), n=d.size))
        if obs == 1:
            per_pair[label] = d.mean(axis=1)
io = np.linalg.norm(P1 - P2, axis=2) * UM
rows.append(dict(method="inter-observer (lower bound)", observer=None, mean_um=io.mean(),
                 sd_um=io.std(ddof=1), median_um=np.median(io),
                 p95_um=np.percentile(io, 95), n=io.size))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(RES, "tre_summary.csv"), index=False)
pd.DataFrame(per_pair, index=[f"{a}-{b}" for a, b in zip(names[:-1], names[1:])]
             ).to_csv(os.path.join(RES, "tre_per_pair.csv"))
print(f"\nregistration took {time.time() - t0:.0f} s")
print("\nTRE between consecutive sections [um]  (n = hole pairs)")
print(summ.round(1).to_string(index=False))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(8.5, 4.2))
for label, v in per_pair.items():
    ax.semilogy(range(1, len(v) + 1), v, "o-", ms=4, label=label)
ax.axhline(np.median(io), color="k", ls=":", label="inter-observer (median)")
ax.set_xlabel("section pair k, k+1"); ax.set_ylabel("mean TRE of 4 holes [µm]")
ax.set_title(f"CODA-style registration ({MODE}), {n} liver sections, observer 1")
ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(RES, "tre_per_pair.png"), dpi=150)

# ------------------------------------------------------------ save images, reslice, transforms
y_cut = int(np.mean(P1[c_idx][:, 1]))
rows_raw, rows_g, rows_e = [], [], []
for k in range(n):
    cg = cv2.warpAffine(colour[k], Tg[k][:2], (W, H), borderValue=(255, 255, 255))
    ce = warp_elastic(cg, fields[k])
    cv2.imwrite(os.path.join(RES, "registered_global", names[k] + ".png"), cg)
    cv2.imwrite(os.path.join(RES, "registered_elastic", names[k] + ".png"), ce)
    rows_raw.append(colour[k][y_cut]); rows_g.append(cg[y_cut]); rows_e.append(ce[y_cut])
Z = 6
gap = np.full((15, W, 3), 255, np.uint8)
stack = [np.repeat(np.stack(r), Z, axis=0) for r in (rows_raw, rows_g, rows_e)]
cv2.imwrite(os.path.join(RES, "reslice.png"),
            np.vstack([stack[0], gap, stack[1], gap, stack[2]]))
np.savez_compressed(os.path.join(RES, "transforms.npz"),
                    names=np.array(names), reference=names[c_idx], um_per_px=UM, pad=pad,
                    global_T=np.stack(Tg), fields=np.stack(fields).astype(np.float16), mode=MODE)
print("results written to", RES, "\nreslice.png: unregistered / global / global+elastic")
