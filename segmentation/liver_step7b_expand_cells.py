import os, re, csv, argparse
import numpy as np
import cv2
from skimage.segmentation import expand_labels
from skimage.morphology import remove_small_objects

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True)
ap.add_argument("--src", default="vitaminp_t03", help="folder written by step 7a")
ap.add_argument("--sections", type=int, nargs="*", default=None)
ap.add_argument("--grow", type=float, nargs="+", default=[6.0], help="maximum growth in um")
ap.add_argument("--mpp", type=float, default=0.46)
ap.add_argument("--white", type=int, default=215, help="grey level above which a pixel is lumen")
ap.add_argument("--red", type=float, default=0.62, help="blue/red ratio below which a pixel is blood")
a = ap.parse_args()

src = os.path.join(a.pth, a.src)
nums = sorted(int(m.group(1)) for f in os.listdir(src)
              if (m := re.fullmatch(r"cells_(\d+)\.png", f)))
if a.sections:
    nums = [n for n in nums if n in a.sections]


def tissue_mask(img):
    b, g, r = [c.astype(np.float32) for c in cv2.split(cv2.GaussianBlur(img, (0, 0), 1.5))]
    grey = (r + g + b) / 3
    lumen = grey > a.white
    blood = (b / (r + 1)) < a.red
    m = ~(lumen | blood)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return remove_small_objects(m, 200)


um2 = a.mpp ** 2
for grow in a.grow:
    dst = os.path.join(src, f"expanded_g{grow:g}")
    os.makedirs(dst, exist_ok=True)
    rows = []
    for n in nums:
        img = cv2.imread(os.path.join(a.pth, f"{n:03d}.png"))
        seeds = cv2.imread(os.path.join(src, f"cells_{n:03d}.png"), cv2.IMREAD_UNCHANGED).astype(np.int32)
        tissue = tissue_mask(img)
        lab = expand_labels(seeds, distance=grow / a.mpp)
        lab[~tissue & (seeds == 0)] = 0
        cv2.imwrite(os.path.join(dst, f"cells_{n:03d}.png"), lab.astype(np.uint16))
        ov = img.copy()
        ov[~tissue] = (0.6 * ov[~tissue] + 0.4 * np.array([255, 255, 0])).astype(np.uint8)  # excluded: cyan tint
        edge = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        ov[edge] = (0, 200, 0)
        cv2.imwrite(os.path.join(dst, f"overlay_{n:03d}.png"), ov)
        area = np.bincount(lab.ravel())[1:]; area = area[area > 0] * um2
        cov = (lab > 0)[tissue].mean()
        rows.append(dict(section=n, grow_um=grow, n_cells=area.size,
                         median_cell_area_um2=round(float(np.median(area)), 1),
                         cells_cover_tissue=round(float(cov), 3), tissue_fraction=round(float(tissue.mean()), 3)))
        print(f"grow {grow:g} um, {n:03d}: {area.size} cells, median {rows[-1]['median_cell_area_um2']} um2 "
              f"(diameter ~{2 * np.sqrt(np.median(area) / np.pi):.0f} um), cells cover {cov:.0%} of the tissue")
    with open(os.path.join(dst, "stats.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
print("check expanded_g*/overlay_NNN.png (cyan = excluded lumen / blood)")
