import os, re, csv, argparse
import numpy as np
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True)
ap.add_argument("--skip", type=int, default=1, help="images generated between two real sections")
ap.add_argument("--row", type=int, default=None, help="y row for the x-z reslice (default: middle)")
ap.add_argument("--dz", type=float, default=5.0, help="real section spacing in um")
ap.add_argument("--px", type=float, default=1.84, help="pixel size in um")
ap.add_argument("--zscale", type=float, default=3.0,
                help="stretch z in the reslice for visibility (1 = true aspect ratio)")
ap.add_argument("--damaged", type=int, nargs="*", default=[5, 6, 12, 13, 14, 28])
a = ap.parse_args()

real = {}
for f in os.listdir(a.pth):
    m = re.fullmatch(r"(\d+)\.(png|jpg|tif)", f, re.I)
    if m:
        real[int(m.group(1))] = os.path.join(a.pth, f)
nums = sorted(real)
gdir = os.path.join(a.pth, f"int_{a.skip}_no_skip")
if not os.path.isdir(gdir):
    raise SystemExit(f"{gdir} not found - run InterpolAI no_skip with --skip {a.skip} first")
gen = {}
for f in os.listdir(gdir):
    m = re.fullmatch(r"(\d+)_int(\d+)\.(png|jpg|tif)", f, re.I)
    if m:
        gen[(int(m.group(1)), int(m.group(2)))] = os.path.join(gdir, f)
print(f"{len(nums)} real sections, {len(gen)} generated images in {gdir}")


def rd(p):
    return imageio.imread(p)[..., :3]


k = a.skip
dst = os.path.join(a.pth, f"stack_skip{k}")
os.makedirs(dst, exist_ok=True)
rows, stack, blend = [], [], []
missing = 0
for i, s in enumerate(nums):
    A = rd(real[s])
    stack.append(A); blend.append(A)
    rows.append(dict(index=len(rows) + 1, z_um=round((s - nums[0]) * a.dz, 3), type="real",
                     sections=f"{s:03d}", damaged=int(s in a.damaged)))
    if i + 1 == len(nums):
        break
    s2 = nums[i + 1]
    B = rd(real[s2]) if s2 == s + 1 else None
    for j in range(1, k + 1):
        t = j / (k + 1)
        p = gen.get((s, j))
        if p is None or B is None:
            missing += 1
            continue
        stack.append(rd(p))
        blend.append(((1 - t) * A.astype(np.float32) + t * B.astype(np.float32) + 0.5).astype(np.uint8))
        rows.append(dict(index=len(rows) + 1, z_um=round((s - nums[0] + t) * a.dz, 3), type="interpolai",
                         sections=f"{s:03d}-{s2:03d}", damaged=int(bool({s, s2} & set(a.damaged)))))
if missing:
    print(f"WARNING: {missing} generated images missing (gaps in the numbering or not generated)")

for r, img in zip(rows, stack):
    imageio.imwrite(os.path.join(dst, f"{r['index']:04d}.png"), img)
with open(os.path.join(dst, "stack_index.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
S = np.stack(stack)
np.save(os.path.join(dst, "stack.npy"), S)
dz_new = a.dz / (k + 1)
print(f"stack: {S.shape[0]} images ({sum(r['type'] == 'real' for r in rows)} real), "
      f"z-spacing {dz_new:g} um, {S.shape[1]} x {S.shape[2]} px at {a.px} um/px -> {dst}")

# ---------------------------------------------------------------- x-z reslice
y = a.row if a.row is not None else S.shape[1] // 2
is_real = np.array([r["type"] == "real" for r in rows])


def xz(vol, dz):
    """one row of every image, stretched in z to the physical aspect ratio"""
    cut = np.stack([v[y] for v in vol])                      # (z, x, 3)
    h = int(round(len(vol) * dz / a.px * a.zscale))         # physical height x zscale
    idx = np.minimum((np.arange(h) * a.px / (dz * a.zscale)).astype(int), len(vol) - 1)
    return cut[idx]


panels = [xz([s for s, r in zip(stack, is_real) if r], a.dz),   # real only, 5 um
          xz(stack, dz_new),                                    # real + InterpolAI
          xz(blend, dz_new)]                                    # real + linear blend
H = max(p.shape[0] for p in panels)
panels = [np.pad(p, ((0, H - p.shape[0]), (0, 0), (0, 0)), constant_values=255) for p in panels]
gap = np.full((H, 12, 3), 255, np.uint8)
imageio.imwrite(os.path.join(dst, "reslice_xz.png"),
                np.concatenate([panels[0], gap, panels[1], gap, panels[2]], axis=1))
print(f"reslice_xz.png (row y = {y}): left real only ({a.dz:g} um) | middle real + InterpolAI "
      f"({dz_new:g} um) | right real + linear blend ({dz_new:g} um); z runs downwards")
