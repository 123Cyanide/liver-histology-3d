import os, sys, argparse, time
import numpy as np
import pandas as pd
import cv2
import tifffile


def fid_path(out, obs):
    """fiducials of step 1: liver_out/step1/ (new layout) or liver_out/ (old layout)"""
    for p in (os.path.join(out, "step1", f"fiducials_observer{obs}.csv"),
              os.path.join(out, f"fiducials_observer{obs}.csv")):
        if os.path.exists(p):
            return p
    sys.exit(f"fiducials_observer{obs}.csv not found in {out} - run step 1 first, "
             f"and run this script from the folder that contains liver_out")



ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--reg", default="coda_python_paper_tile1.5_spacing1.5")
ap.add_argument("--factor", type=int, default=4)
ap.add_argument("--roi", type=int, nargs=4, default=None)
ap.add_argument("--exclude", type=int, nargs="*", default=[])
ap.add_argument("--jpg", action="store_true")
a = ap.parse_args()

npz = np.load(os.path.join(a.out, a.reg, "transforms.npz"))
names = [str(x) for x in npz["names"]]
TG = npz["global_T"].astype(np.float64)
FIELDS = npz["fields"]                                  # (n, H, W, 2) float16, working px
pad, UMW = int(npz["pad"]), float(npz["um_per_px"])
n, H, W = FIELDS.shape[0], FIELDS.shape[1], FIELDS.shape[2]
um_out = 0.46 * a.factor
s = UMW / um_out                                        # working px -> output px (4 for F=4)
Ho, Wo = int(round(H * s)), int(round(W * s))
x0, y0, wr, hr = a.roi if a.roi else (0, 0, Wo, Ho)
wr, hr = min(wr, Wo - x0), min(hr, Ho - y0)
tag = f"highres_f{a.factor}" + (f"_roi{x0}_{y0}_{wr}_{hr}" if a.roi else "")
dst = os.path.join(a.out, tag)
os.makedirs(dst, exist_ok=True)
print(f"{n} sections, frame {W}x{H} @ {UMW:.2f} um -> {Wo}x{Ho} @ {um_out:.2f} um/px; "
      f"writing {wr}x{hr} px region to {dst}")

# output pixel grid (ROI only) expressed in step-3 frame coordinates
xs, ys = np.meshgrid(np.arange(x0, x0 + wr, dtype=np.float32),
                     np.arange(y0, y0 + hr, dtype=np.float32))
xf = (xs + 0.5) / s - 0.5
yf = (ys + 0.5) / s - 0.5


# find the full-res TIFs anywhere below <Data_to_IDA> (folder layout may differ)
TIF = {}
for root, _, fs in os.walk(a.data):
    for f in fs:
        stem, ext = os.path.splitext(f)
        if ext.lower() in (".tif", ".tiff") and "liver" in root.lower() + stem.lower():
            TIF.setdefault(stem, os.path.join(root, f))
missing = [nm for nm in names if nm not in TIF]
if missing:
    sys.exit(f"full-res TIFs not found for {missing[:5]}... under {a.data}\n"
             f"found {len(TIF)} liver TIFs, e.g. {list(TIF.values())[:3]}")
print("full-res images in", os.path.dirname(TIF[names[0]]))


def tif_path(name):
    return TIF[name]


def full_size(name):
    with tifffile.TiffFile(tif_path(name)) as tf:
        return tf.pages[0].shape[:2]                    # (h, w)

hF, wF = full_size(names[0])
w0, h0 = (wF // 8) // 2, (hF // 8) // 2                 # unpadded working size used in step 3
fid = {}
for obs in (1, 2):
    df = pd.read_csv(fid_path(a.out, obs), dtype={"name": str})
    df["name"] = df["name"].str.zfill(3)
    fid[obs] = df.set_index("name")

rows, P_out = [], {1: [], 2: []}
t0 = time.time()
for k, name in enumerate(names):
    T, field = TG[k], FIELDS[k].astype(np.float32)
    Ti = np.linalg.inv(T)
    # ---- points: full res -> working -> global -> elastic -> output grid
    for obs in (1, 2):
        r = fid[obs].loc[name]
        p = np.array([[r[f"x{i}"], r[f"y{i}"]] for i in range(1, 5)], float)  # full-res px
        q = (p + 0.5) * (w0 / wF) - 0.5 + pad                                   # padded working
        g = (np.c_[q, np.ones(4)] @ T.T)[:, :2]
        x = g.copy()
        for _ in range(10):                                                      # x + u(x) = g
            ix = np.clip(np.round(x[:, 0]).astype(int), 0, W - 1)
            iy = np.clip(np.round(x[:, 1]).astype(int), 0, H - 1)
            x = g - field[iy, ix]
        P_out[obs].append((x + 0.5) * s - 0.5)
    if (k + 1) in a.exclude:
        print(f"  {name}: excluded (left for InterpolAI to regenerate)")
        continue
    out_f = os.path.join(dst, name + (".jpg" if a.jpg else ".png"))
    if os.path.exists(out_f):
        continue
    # ---- image: backward map for every output pixel
    # sample the field only at the ROI pixels (same as upsampling the whole field and
    # cropping, but needs no full-frame array: at --factor 1 that would be ~6 GB)
    u = cv2.remap(field, xf, yf, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    gx, gy = xf + u[..., 0], yf + u[..., 1]                                      # global frame
    qx = Ti[0, 0] * gx + Ti[0, 1] * gy + Ti[0, 2] - pad                          # unpadded working
    qy = Ti[1, 0] * gx + Ti[1, 1] * gy + Ti[1, 2] - pad
    src = tifffile.imread(tif_path(name))
    if src.ndim == 3 and src.shape[0] in (3, 4):
        src = np.moveaxis(src, 0, -1)
    src = src[..., :3]
    Hs, Ws = int(round(hF / a.factor)), int(round(wF / a.factor))
    src = cv2.resize(src, (Ws, Hs), interpolation=cv2.INTER_AREA)
    mx = ((qx + 0.5) * (Ws / w0) - 0.5).astype(np.float32)
    my = ((qy + 0.5) * (Hs / h0) - 0.5).astype(np.float32)
    img = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255))
    if a.jpg:
        cv2.imwrite(out_f, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(out_f, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    del src, img, mx, my, u, gx, gy, qx, qy
    print(f"  {name} written  [{time.time() - t0:5.0f} s]")

# ---- fiducials in the output grid (whole frame coordinates) and TRE check
for obs in (1, 2):
    P = np.stack(P_out[obs])
    df = pd.DataFrame({"name": names})
    for i in range(4):
        df[f"x{i+1}"], df[f"y{i+1}"] = P[:, i, 0] - x0, P[:, i, 1] - y0   # relative to ROI
    df.to_csv(os.path.join(dst, f"fiducials_registered_observer{obs}.csv"), index=False)
    d = np.linalg.norm(P[1:] - P[:-1], axis=2) * um_out
    rows.append(dict(observer=obs, mean_um=d.mean(), sd_um=d.std(ddof=1),
                     median_um=np.median(d), max_um=d.max(), n=d.size))
chk = pd.DataFrame(rows)
chk.to_csv(os.path.join(dst, "tre_check.csv"), index=False)
print("\nTRE of the mapped laser holes (must match step 3 'global + elastic'):")
print(chk.round(1).to_string(index=False))
