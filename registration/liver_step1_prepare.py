import os, sys, glob
import numpy as np
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "Data_to_IDA"
FACTOR = int(sys.argv[2]) if len(sys.argv) > 2 else 8
PX_UM = 0.46
HERE = os.path.dirname(os.path.abspath(__file__))
# results go to <project>/liver_out; if the scripts live in <project>/scripts, use the parent folder
PROJECT = os.path.dirname(HERE) if os.path.basename(HERE).lower() == "scripts" else HERE
OUT = os.path.join(PROJECT, "liver_out")
os.makedirs(os.path.join(OUT, "step1"), exist_ok=True)   # csv/png results of step 1


# ---------------------------------------------------------------- (a) images
def downsample_stack():
    import tifffile, cv2
    files = sorted(glob.glob(os.path.join(ROOT, "liver", "*.tif")))
    print(f"{len(files)} liver sections found")
    dst = os.path.join(OUT, f"liver_ds{FACTOR}")
    os.makedirs(dst, exist_ok=True)
    shapes = []
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(dst, name + ".png")
        with tifffile.TiffFile(f) as tf:
            shape = tf.pages[0].shape
        shapes.append((name, *shape[:2]))
        if os.path.exists(out):
            continue
        img = tifffile.imread(f)                       # ~1.2 GB uncompressed, fits in 16 GB RAM
        if img.ndim == 3 and img.shape[0] in (3, 4):   # channel-first -> last
            img = np.moveaxis(img, 0, -1)
        img = img[..., :3]
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // FACTOR, h // FACTOR),
                           interpolation=cv2.INTER_AREA)   # area average
        cv2.imwrite(out, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        print(f"  {name}: {w}x{h} px  ->  {small.shape[1]}x{small.shape[0]} px")
        del img, small
    s = pd.DataFrame(shapes, columns=["name", "height_px", "width_px"])
    s["height_mm"] = s.height_px * PX_UM / 1000
    s["width_mm"] = s.width_px * PX_UM / 1000
    s.to_csv(os.path.join(OUT, "step1", "image_sizes.csv"), index=False)
    print(s.describe().round(2))


# ---------------------------------------------------------------- (b) fiducials
def read_fiducials(observer):
    cand = glob.glob(os.path.join(ROOT, f"fiducialcoo*rdinates_liver_observer{observer}.txt"))
    if not cand:
        sys.exit(f"fiducial file for observer {observer} not found")
    rows = []
    with open(cand[0]) as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) < 9:
                continue
            try:
                vals = [float(v) for v in parts[1:9]]
            except ValueError:          # header line
                continue
            rows.append([os.path.splitext(os.path.basename(parts[0]))[0]] + vals)
    cols = ["name"] + [f"{a}{i}" for i in range(1, 5) for a in ("y", "x")]
    df = pd.DataFrame(rows, columns=cols).sort_values("name").reset_index(drop=True)
    for i in range(1, 5):
        for a in ("y", "x"):
            df[f"{a}{i}_um"] = df[f"{a}{i}"] * PX_UM
            df[f"{a}{i}_ds"] = df[f"{a}{i}"] / FACTOR
    df.to_csv(os.path.join(OUT, "step1", f"fiducials_observer{observer}.csv"), index=False)
    print(f"observer {observer}: {len(df)} sections with 4 holes each")
    return df


def holes_um(df):
    """array (n_sections, 4 holes, 2) in um, order (y, x)"""
    return np.stack([df[[f"y{i}_um", f"x{i}_um"]].to_numpy() for i in range(1, 5)], axis=1)


# ---------------------------------------------------------------- (c) errors
def baseline(df1, df2):
    p1, p2 = holes_um(df1), holes_um(df2)
    # error between consecutive sections WITHOUT registration:
    # the same hole should sit at the same (x, y) on every section
    d = np.linalg.norm(p1[1:] - p1[:-1], axis=2)          # (n-1, 4)
    tre = pd.DataFrame({"pair": [f"{a}-{b}" for a, b in zip(df1.name[:-1], df1.name[1:])],
                        "mean_um": d.mean(axis=1), "max_um": d.max(axis=1)})
    tre.to_csv(os.path.join(OUT, "step1", "baseline_tre.csv"), index=False)
    # inter-observer error: same hole, same section, two people
    common = df1.name.isin(df2.name)
    io = np.linalg.norm(p1[common.to_numpy()] - p2[df2.name.isin(df1.name).to_numpy()], axis=2)

    def ms(x):
        return f"{x.mean():.1f} ± {x.std(ddof=1):.1f} um (n = {x.size})"
    print("\nBefore registration, consecutive sections (observer 1):")
    print("  TRE per hole pair:", ms(d.ravel()))
    print("Inter-observer difference (lower bound for any registration):")
    print("  ", ms(io.ravel()))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, len(d) + 1), d.mean(axis=1), "o-")
    ax[0].set_xlabel("section pair k, k+1"); ax[0].set_ylabel("mean hole distance [µm]")
    ax[0].set_title("Unregistered: error between consecutive sections")
    for i in range(4):
        ax[1].plot(p1[:, i, 1] / 1000, p1[:, i, 0] / 1000, ".-", label=f"hole {i+1}")
    ax[1].invert_yaxis(); ax[1].set_aspect("equal")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("y [mm]")
    ax[1].set_title("Hole positions through the stack"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "step1", "baseline_tre.png"), dpi=150)
    print("figures and tables written to", OUT)


if __name__ == "__main__":
    f1, f2 = read_fiducials(1), read_fiducials(2)
    baseline(f1, f2)
    downsample_stack()
