import os, re, csv, argparse
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True, help="folder with the sections 001.png ...")
ap.add_argument("--src", default=os.path.join("vitaminp_t03", "expanded_g6"),
                help="folder (inside --pth) with cells_NNN.png from step 7b")
ap.add_argument("--nuclei", default="vitaminp_t03", help="folder with nuclei_NNN.png from step 7a")
ap.add_argument("--exclude", type=int, nargs="*", default=[5, 6, 12, 13, 14, 28])
ap.add_argument("--min_overlap", type=float, default=0.5,
                help="overlap / area of the smaller of the two profiles; accepts the small caps\n"
                     "where a cell begins or ends, which a plain IoU would reject")
ap.add_argument("--min_iou", type=float, default=0.0, help="optional IoU floor")
ap.add_argument("--score", choices=["area", "iou"], default="iou",
                help="matching maximises the total IoU (default) or the total overlapping area")
ap.add_argument("--max_shift", type=float, default=15.0, help="um; larger shifts are rejected")
ap.add_argument("--no_align", action="store_true")
ap.add_argument("--blur", type=float, default=3.0, help="Gaussian blur (um) before phase correlation:\n"
                "nuclei differ between sections, the larger tissue pattern does not")
ap.add_argument("--mpp", type=float, default=0.46)
ap.add_argument("--dz", type=float, default=5.0)
a = ap.parse_args()

src = os.path.join(a.pth, a.src)
dst = os.path.join(src, "cells3d")
os.makedirs(dst, exist_ok=True)
nums = sorted(int(m.group(1)) for f in os.listdir(src) if (m := re.fullmatch(r"cells_(\d+)\.png", f)))
if not nums:
    raise SystemExit(f"no cells_NNN.png in {src} - run step 7b first")
print(f"{len(nums)} sections with 2D cells ({nums[0]}..{nums[-1]}); excluded: {a.exclude}")


def rd(path, flag=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(path, flag)
    if im is None:
        raise SystemExit(f"cannot read {path}")
    return im


def grey(n):
    g = rd(os.path.join(a.pth, f"{n:03d}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if a.blur > 0:
        g = cv2.GaussianBlur(g, (0, 0), a.blur / a.mpp)
    return g - g.mean()


H, W = rd(os.path.join(src, f"cells_{nums[0]:03d}.png")).shape[:2]
win = cv2.createHanningWindow((W, H), cv2.CV_32F)


def shift_img(img, dx, dy, nearest=True):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------- fine alignment
use = [n for n in nums if n not in a.exclude]
cum = {use[0]: (0.0, 0.0)}
srows = []
n_rej = n_try = 0
for p, n in zip(use[:-1], use[1:]):
    dx = dy = 0.0
    resp = np.nan
    if not a.no_align and n == p + 1:
        (sx, sy), resp = cv2.phaseCorrelate(grey(p), grey(n), win)
        if np.hypot(sx, sy) * a.mpp <= a.max_shift:
            dx, dy = -sx, -sy                         # move section n onto section p
        else:
            n_rej += 1
        n_try += 1
    cx, cy = cum[p]
    cum[n] = (cx + dx, cy + dy) if n == p + 1 else (0.0, 0.0)   # new block after a gap
    srows.append(dict(pair=f"{p:03d}-{n:03d}", dx_um=round(dx * a.mpp, 2), dy_um=round(dy * a.mpp, 2),
                      response=round(float(resp), 3) if resp == resp else ""))
with open(os.path.join(dst, "shifts.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(srows[0])); w.writeheader(); w.writerows(srows)
sh = np.array([[r["dx_um"], r["dy_um"]] for r in srows])
print(f"fine alignment: median |shift| {np.median(np.hypot(sh[:, 0], sh[:, 1])):.1f} um between neighbours, "
      f"{n_rej}/{n_try} shifts rejected (> {a.max_shift:g} um)")

# ---------------------------------------------------------------- link
nucdir = os.path.join(a.pth, a.nuclei)
z_of = {n: i for i, n in enumerate(nums)}
vol = np.zeros((len(nums), H, W), np.uint32)
cells = {}                      # 3D id -> dict
next_id = 1
prev, prev_n, prev_map = None, None, {}
ious_kept = []
for n in nums:
    if n in a.exclude:
        prev, prev_map = None, {}
        continue
    lab = rd(os.path.join(src, f"cells_{n:03d}.png")).astype(np.int64)
    lab = shift_img(lab.astype(np.float32), *cum[n]).astype(np.int64)
    ids = np.unique(lab); ids = ids[ids > 0]
    area = np.bincount(lab.ravel(), minlength=ids.max() + 1 if ids.size else 1)
    npath = os.path.join(nucdir, f"nuclei_{n:03d}.png")
    has_nuc = set()
    if os.path.exists(npath):
        nuc = shift_img(rd(npath).astype(np.float32), *cum[n]) > 0
        has_nuc = set(np.unique(lab[nuc]).tolist()) - {0}
    cur_map = {}
    if prev is not None and prev_n == n - 1 and ids.size:
        both = (prev > 0) & (lab > 0)
        K = int(lab.max()) + 1
        pair, inter = np.unique(prev[both] * K + lab[both], return_counts=True)
        pa, pb = pair // K, pair % K
        pa_ids, pb_ids = np.unique(pa), np.unique(pb)
        ia = {v: i for i, v in enumerate(pa_ids)}; ib = {v: i for i, v in enumerate(pb_ids)}
        iou = np.zeros((len(pa_ids), len(pb_ids)))
        ovm = np.zeros_like(iou)                         # acceptance: overlap / smaller profile
        ovl = np.zeros_like(iou)                         # assignment: overlapping area
        for x, y, c in zip(pa, pb, inter):
            iou[ia[x], ib[y]] = c / (prev_area[x] + area[y] - c)
            ovm[ia[x], ib[y]] = c / min(prev_area[x], area[y])
            ovl[ia[x], ib[y]] = c
        r, c = linear_sum_assignment(-(ovl if a.score == "area" else iou))
        for i, j in zip(r, c):
            if ovm[i, j] >= a.min_overlap and iou[i, j] >= a.min_iou:
                cur_map[pb_ids[j]] = prev_map[pa_ids[i]]
                ious_kept.append(ovm[i, j])
    for l in ids:
        if l not in cur_map:
            cur_map[l] = next_id
            cells[next_id] = dict(first=n, last=n, n_sections=0, nucleus_sections=0, area_px=0)
            next_id += 1
        c = cells[cur_map[l]]
        c["last"] = n; c["n_sections"] += 1; c["area_px"] += int(area[l])
        c["nucleus_sections"] += int(l in has_nuc)
    lut = np.zeros(int(lab.max()) + 1, np.uint32)
    for l, g in cur_map.items():
        lut[l] = g
    vol[z_of[n]] = lut[lab]
    prev, prev_n, prev_map, prev_area = lab, n, cur_map, area
np.save(os.path.join(dst, "labels3d.npy"), vol)

# ---------------------------------------------------------------- block ends -> complete cells
blocks, cur = [], []
for n in nums:
    if n in a.exclude:
        if cur: blocks.append(cur); cur = []
    else:
        cur.append(n)
if cur: blocks.append(cur)
ends = {b[0] for b in blocks} | {b[-1] for b in blocks}
rows = []
for cid, c in cells.items():
    complete = c["first"] not in ends and c["last"] not in ends
    rows.append(dict(id=cid, first=c["first"], last=c["last"], n_sections=c["n_sections"],
                     extent_um=c["n_sections"] * a.dz, nucleus_sections=c["nucleus_sections"],
                     volume_um3=round(c["area_px"] * a.mpp ** 2 * a.dz, 0), complete=int(complete)))
with open(os.path.join(dst, "cells3d.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)

comp = [r for r in rows if r["complete"]]
ns = np.array([r["n_sections"] for r in comp]) if comp else np.array([0])
nuc = np.array([r["nucleus_sections"] > 0 for r in comp]) if comp else np.array([False])
vol_um3 = np.array([r["volume_um3"] for r in comp]) if comp else np.array([0])
hist = np.bincount(ns, minlength=10)[1:10]
lines = [
    f"sections used: {len(use)} in {len(blocks)} continuous blocks {[f'{b[0]}-{b[-1]}' for b in blocks]}",
    f"linked pairs: {len(ious_kept)}, median overlap {np.median(ious_kept) if ious_kept else float('nan'):.2f}",
    f"3D cells: {len(rows)} in total, {len(comp)} complete (not touching a block end)",
    f"complete cells, sections spanned: median {np.median(ns):.0f} "
    f"(= {np.median(ns) * a.dz:.0f} um; hepatocytes ~20-25 um -> 4-5 sections)",
    "  histogram 1..9 sections: " + " ".join(f"{k}:{v}" for k, v in enumerate(hist, 1)),
    f"  single-section cells: {np.mean(ns == 1):.0%} (mostly small caps / fragments)",
    f"  cells spanning >= 2 sections: median {np.median(ns[ns >= 2]) if np.any(ns >= 2) else 0:.0f} sections",
    f"  share of all cell area (all cells) in cells spanning >= 3 sections: "
    f"{sum(r['volume_um3'] for r in rows if r['n_sections'] >= 3) / max(1, sum(r['volume_um3'] for r in rows)):.0%}",
    f"  with a nucleus in at least one section: {nuc.mean():.0%}",
    f"  median volume (cells with >= 3 sections): "
    f"{np.median(vol_um3[ns >= 3]) if np.any(ns >= 3) else float('nan'):.0f} um^3 "
    f"(a 22 um sphere: {np.pi / 6 * 22 ** 3:.0f} um^3)",
]
open(os.path.join(dst, "summary.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---------------------------------------------------------------- pictures
rng = np.random.default_rng(1)
pal = rng.integers(40, 255, (next_id + 1, 3)).astype(np.uint8); pal[0] = 0
rep = max(1, int(round(a.dz / a.mpp)))
for name, cut in (("xz", vol[:, H // 2, :]), ("yz", vol[:, :, W // 2])):
    img = np.repeat(pal[cut], rep, axis=0)
    cv2.imwrite(os.path.join(dst, f"reslice_{name}.png"), img)
c = 256
y0, x0 = H // 2 - c // 2, W // 2 - c // 2
k0 = max(0, len(use) // 2 - 3)
tiles = []
for n in use[k0:k0 + 6]:
    sec = shift_img(rd(os.path.join(a.pth, f"{n:03d}.png"), cv2.IMREAD_COLOR).astype(np.float32),
                    *cum[n], nearest=False).astype(np.uint8)
    lab = vol[z_of[n]]
    col = pal[lab]
    t = np.where(lab[..., None] > 0, (0.55 * sec + 0.45 * col).astype(np.uint8), sec)
    edge = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    t[edge] = 255
    t = t[y0:y0 + c, x0:x0 + c].copy()
    cv2.putText(t, f"{n:03d}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    tiles += [t, np.full((c, 6, 3), 255, np.uint8)]
cv2.imwrite(os.path.join(dst, "sections_crop.png"), np.concatenate(tiles[:-1], axis=1))
print("results in", dst, "(same colour = same 3D cell; reslice z is stretched to scale)")
