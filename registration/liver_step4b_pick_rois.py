import os, sys, glob, argparse
import numpy as np
import pandas as pd
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=3)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--max_white", type=float, default=0.15)
ap.add_argument("--hole_margin", type=int, default=150)
ap.add_argument("--step", type=int, default=128)
ap.add_argument("--max_outside", type=float, default=0.01,
                help="max fraction outside the tissue (white connected to the frame border)")
ap.add_argument("--avoid", type=int, nargs=3, action="append", default=[],
                metavar=("X", "Y", "S"), help="window to keep free (e.g. a test ROI); repeatable")
ap.add_argument("--save", default=None, help="write the picked ROIs to this CSV")
ap.add_argument("--preview", default="roi_candidates.png")
ap.add_argument("--quantile", type=float, default=0.9)
ap.add_argument("--ignore", type=int, nargs="*", default=[5, 6, 12, 13, 14])
a = ap.parse_args()

d = os.path.join(a.out, "highres_f4")
files = sorted(glob.glob(os.path.join(d, "[0-9][0-9][0-9].jpg")) +
               glob.glob(os.path.join(d, "[0-9][0-9][0-9].png")))
if not files:
    sys.exit(f"no registered sections in {d} - run step 4 without --roi first")
D = 16                                              # work on a 16x smaller grid
white, outside = [], []
for f in files:
    g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    H, W = g.shape
    w = cv2.resize((g > 220).astype(np.float32), (W // D, H // D), interpolation=cv2.INTER_AREA)
    # outside the tissue = white region connected to the frame border (also tears that open to it).
    # Done on the coarse grid (29 um cells, > 50 % white): at full resolution the white
    # sinusoids form a network that would connect the whole interior to the outside.
    bw = (w > 0.5).astype(np.uint8)
    n_lab, lab = cv2.connectedComponents(bw, connectivity=4)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    white.append(w)
    outside.append(np.isin(lab, border[border > 0]).astype(np.float32))
white = np.stack(white)                             # (n, H/D, W/D) white fraction per cell
outside = np.stack(outside)                         # (n, H/D, W/D) fraction outside the tissue
nums = [int(os.path.splitext(os.path.basename(f))[0]) for f in files]
use = np.array([k not in a.ignore for k in nums])
print(f"{len(files)} sections, frame {W} x {H} px")

holes = []
fp = os.path.join(d, "fiducials_registered_observer1.csv")
if os.path.exists(fp):
    df = pd.read_csv(fp, dtype={"name": str})
    df = df[[int(nm) not in a.ignore for nm in df["name"]]]   # damaged sections: holes mapped badly
    for i in range(1, 5):
        holes += list(zip(df[f"x{i}"], df[f"y{i}"]))
holes = np.array(holes, float).reshape(-1, 2)

S, cands = a.size, []
best_any = (np.inf, 0, 0)
n_white = n_hole = n_edge = 0
for y in range(0, H - S + 1, a.step):
    for x in range(0, W - S + 1, a.step):
        wf = white[use, y // D:(y + S) // D, x // D:(x + S) // D].mean(axis=(1, 2))
        q = np.quantile(wf, a.quantile)
        best_any = min(best_any, (q, x, y))
        if q > a.max_white:
            n_white += 1
            continue
        ob = outside[use, y // D:(y + S) // D, x // D:(x + S) // D].mean(axis=(1, 2))
        qo = np.quantile(ob, a.quantile)
        if qo > a.max_outside:
            n_edge += 1
            continue
        m = a.hole_margin
        if holes.size and np.any((holes[:, 0] > x - m) & (holes[:, 0] < x + S + m) &
                                 (holes[:, 1] > y - m) & (holes[:, 1] < y + S + m)):
            n_hole += 1
            continue
        cands.append((q, wf.mean(), x, y, qo))
cands.sort()
pick = []
free = lambda c, x, y, s: c[2] + S <= x or x + s <= c[2] or c[3] + S <= y or y + s <= c[3]
for c in cands:
    if all(free(c, p[2], p[3], S) for p in pick) and all(free(c, *av) for av in a.avoid):
        pick.append(c)
    if len(pick) == a.n:
        break
print(f"windows rejected: {n_white} too white, {n_edge} touch the tissue edge, "
      f"{n_hole} too close to a laser hole")
if not pick:
    q, x, y = best_any
    sys.exit(f"no window passes (lowest white fraction {q:.3f} at --roi {x} {y} {S} {S}). "
             f"Relax the criterion that rejects most windows above: --max_white, "
             f"--max_outside, --hole_margin, or use a smaller --size")

ref = cv2.imread(files[len(files) // 2])
for k, (wmax, wmean, x, y, qo) in enumerate(pick, 1):
    cv2.rectangle(ref, (x, y), (x + S, y + S), (0, 160, 0), 25)
    cv2.putText(ref, f"ROI{k}", (x + 30, y + 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 160, 0), 12)
for ax_, ay_, as_ in a.avoid:
    cv2.rectangle(ref, (ax_, ay_), (ax_ + as_, ay_ + as_), (200, 0, 0), 25)
for hx, hy in holes:
    cv2.circle(ref, (int(hx), int(hy)), 60, (0, 0, 255), 12)
cv2.imwrite(os.path.join(d, a.preview), cv2.resize(ref, (W // 4, H // 4)))

print(f"ignored sections {a.ignore}; criteria in {a.quantile:.0%} of the rest: white <= {a.max_white}, "
      f"outside tissue <= {a.max_outside}, laser holes >= {a.hole_margin} px away")
print(f"{len(cands)} windows pass; picked {len(pick)}:")
for k, (wmax, wmean, x, y, qo) in enumerate(pick, 1):
    print(f"  ROI{k}: --roi {x} {y} {S} {S}    white {a.quantile:.0%}-quantile {wmax:.3f}, "
          f"mean {wmean:.3f}; outside tissue {a.quantile:.0%}-quantile {qo:.3f}")
print("preview:", os.path.join(d, a.preview), "(green = ROIs, red = laser holes)")
if a.save:
    pd.DataFrame([dict(roi=k, x=x, y=y, size=S, white_q=round(wmax, 4), outside_q=round(qo, 4))
                  for k, (wmax, wmean, x, y, qo) in enumerate(pick, 1)]).to_csv(a.save, index=False)
    print("ROI list written to", a.save, "(coordinates in the highres_f4 grid, 1.84 um/px)")
