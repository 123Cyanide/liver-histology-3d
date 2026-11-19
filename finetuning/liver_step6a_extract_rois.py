"""
Step 6a: cut many registered ROIs at full resolution (0.46 um/px) in ONE pass over the
47 full-resolution sections (each 1.2 GB TIF is read only once).

ROIs come from step 4b (--save rois.csv, coordinates in the highres_f4 grid); they are
scaled by 4 to the full-resolution grid, so --size 256 in step 4b gives 1024 x 1024 px.

  python liver_step6a_extract_rois.py D:\\master\\Data_to_IDA liver_out liver_out\\train_rois.csv

Output: liver_out/train_rois/roi01/001.png ... 047.png, roi02/..., and rois_fullres.csv
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd
import cv2
import tifffile

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("rois")
ap.add_argument("--reg", default="coda_python_paper_tile1.5_spacing1.5")
ap.add_argument("--from_factor", type=int, default=4, help="grid of the ROI coordinates")
ap.add_argument("--factor", type=int, default=1, help="output resolution 0.46*factor um/px")
ap.add_argument("--dst", default="train_rois")
a = ap.parse_args()

npz = np.load(os.path.join(a.out, a.reg, "transforms.npz"))
names = [str(x) for x in npz["names"]]
TG = npz["global_T"].astype(np.float64)
FIELDS = npz["fields"]
pad, UMW = int(npz["pad"]), float(npz["um_per_px"])
s = UMW / (0.46 * a.factor)
R = pd.read_csv(a.rois)
k = a.from_factor / a.factor
rois = [(int(r.x * k), int(r.y * k), int(r["size"] * k)) for _, r in R.iterrows()]
dst = os.path.join(a.out, a.dst)
os.makedirs(dst, exist_ok=True)
pd.DataFrame([dict(roi=f"roi{i+1:02d}", x=x, y=y, size=S, um_per_px=0.46 * a.factor)
              for i, (x, y, S) in enumerate(rois)]).to_csv(os.path.join(dst, "rois_fullres.csv"), index=False)
grids = []
for i, (x0, y0, S) in enumerate(rois):
    os.makedirs(os.path.join(dst, f"roi{i+1:02d}"), exist_ok=True)
    xs, ys = np.meshgrid(np.arange(x0, x0 + S, dtype=np.float32), np.arange(y0, y0 + S, dtype=np.float32))
    grids.append(((xs + 0.5) / s - 0.5, (ys + 0.5) / s - 0.5))
print(f"{len(rois)} ROIs of {rois[0][2]} px at {0.46 * a.factor:.2f} um/px -> {dst}")

TIF = {}
for root, _, fs in os.walk(a.data):
    for f in fs:
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".tif", ".tiff") and "liver" in root.lower() + stem.lower():
            TIF.setdefault(stem, os.path.join(root, f))
if any(nm not in TIF for nm in names):
    sys.exit(f"full-res TIFs not found under {a.data}")
with tifffile.TiffFile(TIF[names[0]]) as tf_:
    hF, wF = tf_.pages[0].shape[:2]
w0, h0 = (wF // 8) // 2, (hF // 8) // 2
Hs, Ws = int(round(hF / a.factor)), int(round(wF / a.factor))

t0 = time.time()
for n, name in enumerate(names):
    outs = [os.path.join(dst, f"roi{i+1:02d}", name + ".png") for i in range(len(rois))]
    if all(os.path.exists(o) for o in outs):
        continue
    field, Ti = FIELDS[n].astype(np.float32), np.linalg.inv(TG[n])
    src = tifffile.imread(TIF[name])
    if src.ndim == 3 and src.shape[0] in (3, 4):
        src = np.moveaxis(src, 0, -1)
    src = src[..., :3]
    if a.factor != 1:
        src = cv2.resize(src, (Ws, Hs), interpolation=cv2.INTER_AREA)
    for (xf, yf), o in zip(grids, outs):
        u = cv2.remap(field, xf, yf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        gx, gy = xf + u[..., 0], yf + u[..., 1]
        qx = Ti[0, 0] * gx + Ti[0, 1] * gy + Ti[0, 2] - pad
        qy = Ti[1, 0] * gx + Ti[1, 1] * gy + Ti[1, 2] - pad
        mx = ((qx + 0.5) * (Ws / w0) - 0.5).astype(np.float32)
        my = ((qy + 0.5) * (Hs / h0) - 0.5).astype(np.float32)
        img = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255))
        cv2.imwrite(o, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    del src
    print(f"  {name}: {len(rois)} ROIs written  [{time.time() - t0:5.0f} s]")
print("done")
