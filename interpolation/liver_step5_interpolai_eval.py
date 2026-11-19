import os, re, csv, argparse, time
import numpy as np
import imageio

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True)
ap.add_argument("--skip", type=int, nargs="+", default=[1])
ap.add_argument("--damaged", type=int, nargs="*", default=[5, 6, 12, 13, 14, 28])
ap.add_argument("--examples", type=int, default=3)
ap.add_argument("--model", default=os.path.join("interpolation", "model"))
ap.add_argument("--tile", type=int, default=1024)
ap.add_argument("--mock", action="store_true", help="no TensorFlow: use the blend (code test only)")
ap.add_argument("--name", default="eval_interpolai", help="results folder inside --pth "
                "(e.g. eval_finetuned for a fine-tuned model)")
a = ap.parse_args()

# ------------------------------------------------------------------ images
files = {}
for f in os.listdir(a.pth):
    m = re.fullmatch(r"(\d+)\.(png|jpg|tif)", f, re.I)   # only 001.png ..., not montage_..._skip10.png
    if m:
        files[int(m.group(1))] = os.path.join(a.pth, f)
nums = sorted(files)
print(f"{len(nums)} sections ({nums[0]}..{nums[-1]}) in {a.pth}")
cache = {}


def load(k):
    if k not in cache:
        cache[k] = imageio.imread(files[k])[..., :3].astype(np.float32) / 255.0
    return cache[k]


# ------------------------------------------------------------------ model
if a.mock:
    def film(A, B, t):
        return (1 - t) * A + t * B
else:
    import tensorflow as tf
    model = tf.saved_model.load(a.model)

    def film_tile(A, B, t):
        out = model({"time": np.array([[t]], np.float32),
                     "x0": A[None].astype(np.float32), "x1": B[None].astype(np.float32)})
        return np.clip(out["image"][0].numpy(), 0, 1)

    def film(A, B, t):
        T = a.tile
        h, w = A.shape[:2]
        if max(h, w) <= T:
            return film_tile(A, B, t)
        ph, pw = (-h) % T, (-w) % T                      # pad with white, tile, stitch
        Ap = np.pad(A, ((0, ph), (0, pw), (0, 0)), constant_values=1)
        Bp = np.pad(B, ((0, ph), (0, pw), (0, 0)), constant_values=1)
        out = np.ones_like(Ap)
        for y in range(0, Ap.shape[0], T):
            for x in range(0, Ap.shape[1], T):
                out[y:y + T, x:x + T] = film_tile(Ap[y:y + T, x:x + T], Bp[y:y + T, x:x + T], t)
        return out[:h, :w]


# ------------------------------------------------------------------ metrics (tissue only)
def gauss_matrix(n, sigma=1.5, r=5):
    i = np.arange(n)
    G = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma) ** 2) * (np.abs(i[:, None] - i[None, :]) <= r)
    return (G / G.sum(1, keepdims=True)).astype(np.float32)

_G = {}


def blur(x):
    h, w = x.shape
    if h not in _G: _G[h] = gauss_matrix(h)
    if w not in _G: _G[w] = gauss_matrix(w)
    return _G[h] @ x @ _G[w].T


def gray(x):
    return x @ np.array([0.299, 0.587, 0.114], np.float32)


def ssim_map(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    x, y = gray(x), gray(y)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))


def tissue(x):
    return gray(x) < 0.86                       # white background ~ 0.9-1.0


def metrics(pred, real):
    m = tissue(real)
    if m.sum() < 1000:
        return np.nan, np.nan
    mse = np.mean((pred[m] - real[m]) ** 2)
    return 10 * np.log10(1.0 / max(mse, 1e-10)), float(ssim_map(pred, real)[m].mean())


# ------------------------------------------------------------------ run
dst = os.path.join(a.pth, a.name)
os.makedirs(os.path.join(dst, "examples"), exist_ok=True)
os.makedirs(os.path.join(dst, "film"), exist_ok=True)
bad = set(a.damaged)
rows = []
t0 = time.time()
for k in a.skip:
    n_ex = 0
    for A_n in nums:
        B_n = A_n + k + 1
        targets = list(range(A_n + 1, B_n))
        if B_n not in files or any(t not in files for t in targets):
            continue
        A, B = load(A_n), load(B_n)
        for j, tg in enumerate(targets, 1):
            t = j / (k + 1)
            real = load(tg)
            pf = film(A, B, t)
            pb = (1 - t) * A + t * B
            pn = A if t <= 0.5 else B
            clean = not ({A_n, tg, B_n} & bad)
            r = dict(skip=k, A=A_n, target=tg, B=B_n, t=round(t, 3), clean=int(clean))
            for lab, p in (("film", pf), ("blend", pb), ("nearest", pn)):
                r[f"psnr_{lab}"], r[f"ssim_{lab}"] = metrics(p, real)
            rows.append(r)
            imageio.imwrite(os.path.join(dst, "film", f"{tg:03d}_skip{k}.png"),
                            (pf * 255 + 0.5).astype(np.uint8))
            if clean and n_ex < a.examples and j == (k + 1) // 2:
                panel = np.concatenate([A, real, pf, pb, B], axis=1)[::2, ::2]
                imageio.imwrite(os.path.join(dst, "examples",
                                f"skip{k}_{A_n:03d}-{tg:03d}-{B_n:03d}_A_real_FILM_blend_B.png"),
                                (panel * 255 + 0.5).astype(np.uint8))
                n_ex += 1
            print(f"skip {k}: {A_n:03d} -> [{tg:03d}] <- {B_n:03d}  t={t:.2f}  "
                  f"SSIM film {r['ssim_film']:.3f} / blend {r['ssim_blend']:.3f}"
                  f"{'' if clean else '  (damaged)'}  [{time.time() - t0:.0f} s]")
        for c in [c for c in cache if c < A_n]:          # keep memory small
            del cache[c]

cols = list(rows[0].keys())
with open(os.path.join(dst, "per_frame.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)

summ = []
for k in a.skip:
    for subset in ("all", "clean"):
        R = [r for r in rows if r["skip"] == k and (subset == "all" or r["clean"])]
        if not R:
            continue
        s = dict(skip=k, subset=subset, n=len(R))
        for met in ("ssim", "psnr"):
            for lab in ("film", "blend", "nearest"):
                v = np.array([r[f"{met}_{lab}"] for r in R], float)
                v = v[~np.isnan(v)]
                s[f"{met}_{lab}_mean"] = round(float(v.mean()), 4)
                s[f"{met}_{lab}_sd"] = round(float(v.std(ddof=1)), 4) if v.size > 1 else np.nan
                s[f"{met}_{lab}_median"] = round(float(np.median(v)), 4)
        summ.append(s)
with open(os.path.join(dst, "summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(summ[0].keys())); w.writeheader(); w.writerows(summ)

print("\nSSIM / PSNR on tissue pixels, mean ± SD   (FILM | linear blend | nearest input)")
for s in summ:
    print(f"skip {s['skip']} {s['subset']:5s} n={s['n']:3d}  "
          f"SSIM {s['ssim_film_mean']:.3f}±{s['ssim_film_sd']:.3f} | "
          f"{s['ssim_blend_mean']:.3f}±{s['ssim_blend_sd']:.3f} | "
          f"{s['ssim_nearest_mean']:.3f}±{s['ssim_nearest_sd']:.3f}   "
          f"PSNR {s['psnr_film_mean']:.1f} | {s['psnr_blend_mean']:.1f} | {s['psnr_nearest_mean']:.1f} dB")
print("results in", dst)
