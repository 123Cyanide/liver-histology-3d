import os, re, csv, json, glob, argparse, time
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True, help="folder with 001.png ... (registered sections)")
ap.add_argument("--sections", type=int, nargs="*", default=None, help="only these sections")
ap.add_argument("--model", default="flex", help="VitaminP model: flex (large) or syn")
ap.add_argument("--mpp", type=float, default=0.46, help="pixel size of the images in um")
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--threshold", type=float, default=0.5, help="detection threshold")
ap.add_argument("--cell_threshold", type=float, default=0.5, help="lower -> larger cells")
ap.add_argument("--out", default="vitaminp", help="results folder inside --pth")
ap.add_argument("--min_area", type=float, default=10.0, help="minimum object area in um^2")
ap.add_argument("--device", default="cuda")
ap.add_argument("--parse_only", action="store_true", help="only convert existing raw/ GeoJSON")
a = ap.parse_args()

files = {}
for f in os.listdir(a.pth):
    m = re.fullmatch(r"(\d+)\.(png|jpg|tif)", f, re.I)
    if m:
        files[int(m.group(1))] = os.path.join(a.pth, f)
nums = sorted(n for n in files if a.sections is None or n in a.sections)
dst = os.path.join(a.pth, a.out)
os.makedirs(dst, exist_ok=True)
print(f"{len(nums)} sections from {a.pth}")

# ------------------------------------------------------------- run VitaminP
if not a.parse_only:
    import vitaminp
    from vitaminp.inference import WSIPredictor
    model = vitaminp.load_model(a.model, device=a.device)
    predictor = WSIPredictor(model=model, device=a.device, patch_size=512, overlap=64,
                             target_mpp=a.mpp, magnification=20, batch_size=a.batch,
                             cell_threshold=a.cell_threshold)
    t0 = time.time()
    for n in nums:
        out = os.path.join(dst, "raw", f"{n:03d}")
        if glob.glob(os.path.join(out, "**", "*cell*segmentation.geojson"), recursive=True):
            continue
        res = predictor.predict(wsi_path=files[n], output_dir=out, branches=["he_nuclei", "he_cell"],
                                filter_tissue=False, clean_overlaps=True, save_geojson=True,
                                save_visualization=False, detection_threshold=a.threshold,
                                min_area_um=a.min_area, mpp_override=a.mpp)
        print(f"  {n:03d}: {res.get('he_cell', {}).get('num_detections', '?')} cells, "
              f"{res.get('he_nuclei', {}).get('num_detections', '?')} nuclei  [{time.time() - t0:.0f} s]")


# ------------------------------------------------------------- GeoJSON -> label images
def polygons(geom):
    """all non-empty polygons of a GeoJSON geometry (empty ones come from shapely's buffer(0))"""
    if not geom:
        return []
    t = geom.get("type")
    if t == "Polygon":
        out = [geom.get("coordinates") or []]
    elif t == "MultiPolygon":
        out = geom.get("coordinates") or []
    elif t == "GeometryCollection":
        out = [p for g in geom.get("geometries", []) for p in polygons(g)]
    else:
        out = []
    return [p for p in out if p and len(p[0]) >= 3]


def rasterize(path, shape):
    lab = np.zeros(shape, np.uint16)
    feats = json.load(open(path))["features"] if path else []
    outside = 0
    for i, f in enumerate(feats, 1):
        for poly in polygons(f["geometry"]):
            ext = np.round(np.array(poly[0], float)).astype(np.int32)
            if (ext[:, 0].max() < 0 or ext[:, 1].max() < 0 or
                    ext[:, 0].min() >= shape[1] or ext[:, 1].min() >= shape[0]):
                outside += 1
                continue
            cv2.fillPoly(lab, [ext], i)
            for hole in poly[1:]:
                cv2.fillPoly(lab, [np.round(np.array(hole, float)).astype(np.int32)], 0)
    if outside:
        print(f"    WARNING {outside} of {len(feats)} polygons lie outside the image - coordinate scale?")
    return lab


def find(n, kind):
    g = glob.glob(os.path.join(dst, "raw", f"{n:03d}", "**", f"*{kind}*segmentation.geojson"), recursive=True)
    return g[0] if g else None


rows = []
um2 = a.mpp ** 2
for n in nums:
    img = cv2.imread(files[n])
    cpath, npath = find(n, "cell"), find(n, "nuclei")
    if cpath is None and npath is None:
        print(f"  {n:03d}: no VitaminP output")
        continue
    cells, nuc = rasterize(cpath, img.shape[:2]), rasterize(npath, img.shape[:2])
    cv2.imwrite(os.path.join(dst, f"cells_{n:03d}.png"), cells)
    cv2.imwrite(os.path.join(dst, f"nuclei_{n:03d}.png"), nuc)
    ov = img.copy()
    for lab, col in ((cells, (0, 200, 0)), (nuc, (255, 80, 0))):
        edge = cv2.morphologyEx(lab.astype(np.float32), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        ov[edge] = col
    cv2.imwrite(os.path.join(dst, f"overlay_{n:03d}.png"), ov)
    ca = np.bincount(cells.ravel())[1:]; ca = ca[ca > 0] * um2
    na = np.bincount(nuc.ravel())[1:]; na = na[na > 0] * um2
    tissue = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 220).mean()
    rows.append(dict(section=n, n_cells=ca.size, n_nuclei=na.size,
                     median_cell_area_um2=round(float(np.median(ca)), 1) if ca.size else "",
                     median_nucleus_area_um2=round(float(np.median(na)), 1) if na.size else "",
                     cell_coverage=round(float((cells > 0).mean()), 3), tissue_fraction=round(float(tissue), 3)))
    r = rows[-1]
    print(f"  {n:03d}: {r['n_cells']} cells (median {r['median_cell_area_um2']} um2), "
          f"{r['n_nuclei']} nuclei (median {r['median_nucleus_area_um2']} um2), "
          f"cells cover {r['cell_coverage']:.0%} of the image")
if rows:
    with open(os.path.join(dst, "stats.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    print("results in", dst, "- check overlay_NNN.png")
