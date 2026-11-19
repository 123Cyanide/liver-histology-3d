import os, re, csv, argparse
import numpy as np
import imageio.v2 as imageio
from scipy import ndimage as ndi

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True)
ap.add_argument("--skip", type=int, nargs="+", default=[1, 3])
ap.add_argument("--damaged", type=int, nargs="*", default=[5, 6, 12, 13, 14, 28],
                help="known damaged sections (added to the automatic flag)")
ap.add_argument("--px", type=float, default=1.84, help="pixel size of the input images in um")
ap.add_argument("--down", type=int, default=4, help="averaging factor (4 -> 7.36 um/px)")
ap.add_argument("--white", type=float, default=0.80, help="grey level above which a pixel is lumen")
ap.add_argument("--tear", type=float, default=0.05, help="extra border-white fraction = tear")
ap.add_argument("--name", default="eval_interpolai", help="results folder inside --pth "
                "(e.g. eval_finetuned for a fine-tuned model)")
a = ap.parse_args()

files = {}
for f in os.listdir(a.pth):
    m = re.fullmatch(r"(\d+)\.(png|jpg|tif)", f, re.I)   # only 001.png ..., not montage_..._skip10.png
    if m:
        files[int(m.group(1))] = os.path.join(a.pth, f)
nums = sorted(files)
ev = os.path.join(a.pth, a.name)
W = np.array([0.299, 0.587, 0.114], np.float32)


def small_gray(path):
    x = imageio.imread(path)[..., :3].astype(np.float32) / 255.0 @ W
    d = a.down
    h, w = (x.shape[0] // d) * d, (x.shape[1] // d) * d
    return x[:h, :w].reshape(h // d, d, w // d, d).mean((1, 3))


def background(g):
    """white regions touching the ROI border with > 2 % of the area (outside tissue / tears)"""
    lab, n = ndi.label(g > a.white)
    if n == 0:
        return np.zeros_like(g, bool)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    sizes = ndi.sum(np.ones_like(g), lab, index=np.arange(1, n + 1))
    keep = [l for l in border if l > 0 and sizes[l - 1] > 0.02 * g.size]
    return np.isin(lab, keep)


def lumen(g, bg):
    m = (g > a.white) & ~bg
    return ndi.binary_opening(m, iterations=1)       # drop single-pixel noise


def dice(p, q):
    s = p.sum() + q.sum()
    return np.nan if s == 0 else 2.0 * (p & q).sum() / s


def ssim(x, y, m, C1=0.01 ** 2, C2=0.03 ** 2, s=1.5):
    f = lambda z: ndi.gaussian_filter(z, s)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx * mx, f(y * y) - my * my, f(x * y) - mx * my
    S = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return float(S[m].mean())


# ---------------------------------------------------------------- damage per section
G = {k: small_gray(files[k]) for k in nums}
BG = {k: background(G[k]) for k in nums}
frac = {k: BG[k].mean() for k in nums}
med = np.median(list(frac.values()))
auto = {k for k in nums if frac[k] > med + a.tear}
bad = auto | set(a.damaged)
with open(os.path.join(ev, "damage_per_section.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["section", "border_white_fraction", "auto_tear", "in_list"])
    for k in nums:
        w.writerow([k, round(frac[k], 3), int(k in auto), int(k in a.damaged)])
print(f"median border-white fraction {med:.3f}; automatically flagged tears: {sorted(auto)}")
print(f"excluded from 'clean': {sorted(bad)}")

# ---------------------------------------------------------------- per frame
rows = []
for k in a.skip:
    for A_n in nums:
        B_n = A_n + k + 1
        tg_list = list(range(A_n + 1, B_n))
        if B_n not in files or any(t not in files for t in tg_list):
            continue
        for j, tg in enumerate(tg_list, 1):
            fp = os.path.join(ev, "film", f"{tg:03d}_skip{k}.png")
            if not os.path.exists(fp):
                continue
            t = j / (k + 1)
            real = G[tg]
            preds = {"film": small_gray(fp),
                     "blend": (1 - t) * G[A_n] + t * G[B_n],
                     "nearest": G[A_n] if t <= 0.5 else G[B_n]}
            region = ~BG[tg]                                   # tissue + inner lumens of the real section
            L_real = lumen(real, BG[tg])
            r = dict(skip=k, A=A_n, target=tg, B=B_n, t=round(t, 3),
                     clean=int(not ({A_n, tg, B_n} & bad)), lumen_px_real=int(L_real.sum()))
            for lab, p in preds.items():
                r[f"ssim74_{lab}"] = round(ssim(p, real, region), 4)
                r[f"dice_{lab}"] = round(dice(lumen(p, BG[tg]), L_real), 4)
            rows.append(r)
if not rows:
    raise SystemExit("no FILM images found - run liver_step5_interpolai_eval.py first")

with open(os.path.join(ev, "structure_per_frame.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)

summ = []
for k in a.skip:
    for subset in ("all", "clean"):
        R = [r for r in rows if r["skip"] == k and (subset == "all" or r["clean"])]
        if len(R) < 2:
            continue
        s = dict(skip=k, subset=subset, n=len(R))
        for met in ("ssim74", "dice"):
            for lab in ("film", "blend", "nearest"):
                v = np.array([r[f"{met}_{lab}"] for r in R], float)
                v = v[~np.isnan(v)]
                s[f"{met}_{lab}_mean"], s[f"{met}_{lab}_sd"] = round(v.mean(), 3), round(v.std(ddof=1), 3)
        # paired: in how many frames is FILM better than the blend?
        s["dice_film_better_than_blend"] = f"{sum(r['dice_film'] > r['dice_blend'] for r in R)}/{len(R)}"
        summ.append(s)
with open(os.path.join(ev, "structure_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(summ[0])); w.writeheader(); w.writerows(summ)

print(f"\nstructure scale ({a.px * a.down:.2f} um/px), mean ± SD   (FILM | linear blend | nearest)")
for s in summ:
    print(f"skip {s['skip']} {s['subset']:5s} n={s['n']:3d}  "
          f"SSIM {s['ssim74_film_mean']:.3f}±{s['ssim74_film_sd']:.3f} | {s['ssim74_blend_mean']:.3f}±{s['ssim74_blend_sd']:.3f} | "
          f"{s['ssim74_nearest_mean']:.3f}±{s['ssim74_nearest_sd']:.3f}   "
          f"lumen Dice {s['dice_film_mean']:.3f}±{s['dice_film_sd']:.3f} | {s['dice_blend_mean']:.3f}±{s['dice_blend_sd']:.3f} | "
          f"{s['dice_nearest_mean']:.3f}±{s['dice_nearest_sd']:.3f}   FILM>blend {s['dice_film_better_than_blend']}")
print("results in", ev)
