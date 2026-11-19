import os, argparse
import numpy as np
import imageio.v2 as imageio

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True)
ap.add_argument("--section", type=int, required=True)
ap.add_argument("--skip", type=int, default=10)
ap.add_argument("--crop", type=int, default=None)
ap.add_argument("--dz", type=float, default=5.0)
a = ap.parse_args()

A, B, k = a.section, a.section + 1, a.skip
gdir = os.path.join(a.pth, f"int_{k}_no_skip")


def find(folder, stem):
    for ext in (".png", ".jpg", ".tif"):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    raise SystemExit(f"{stem} not found in {folder}")


paths = [find(a.pth, f"{A:03d}")] + [find(gdir, f"{A:03d}_int{j}") for j in range(1, k + 1)] \
        + [find(a.pth, f"{B:03d}")]
imgs = [imageio.imread(p)[..., :3] for p in paths]
if a.crop:
    h, w = imgs[0].shape[:2]
    y0, x0 = (h - a.crop) // 2, (w - a.crop) // 2
    imgs = [im[y0:y0 + a.crop, x0:x0 + a.crop] for im in imgs]
z = [0.0] + [a.dz * j / (k + 1) for j in range(1, k + 1)] + [a.dz]
labels = [f"real {A:03d}"] + [f"InterpolAI {j}" for j in range(1, k + 1)] + [f"real {B:03d}"]
dst = os.path.join(a.pth, f"montage_{A:03d}_{B:03d}_skip{k}.png")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(imgs)
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows))
    for ax in np.atleast_1d(axs).ravel():
        ax.axis("off")
    for ax, im, lab, zz in zip(np.atleast_1d(axs).ravel(), imgs, labels, z):
        ax.imshow(im)
        ax.set_title(f"{lab}\nz = {zz:.2f} µm", fontsize=9,
                     color="black" if lab.startswith("real") else "tab:blue")
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
except ImportError:
    gap = np.full((imgs[0].shape[0], 8, 3), 255, np.uint8)
    strip = [imgs[0]]
    for im in imgs[1:]:
        strip += [gap, im]
    imageio.imwrite(dst, np.concatenate(strip, axis=1))
print("written", dst)
for lab, zz in zip(labels, z):
    print(f"  {lab:14s} z = {zz:.2f} um")
