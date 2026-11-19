import os, sys, csv
import numpy as np
from scipy.stats import wilcoxon

rois = sys.argv[1:]
rows = []
for r in rois:
    with open(os.path.join(r, "eval_interpolai", "structure_per_frame.csv")) as fh:
        for row in csv.DictReader(fh):
            row["roi"] = os.path.basename(os.path.normpath(r))
            rows.append(row)
print(f"{len(rows)} frames from {len(rois)} ROIs")

out = []
for k in sorted({int(r["skip"]) for r in rows}):
    for subset in ("all", "clean"):
        R = [r for r in rows if int(r["skip"]) == k and (subset == "all" or r["clean"] == "1")]
        d = {lab: np.array([float(r[f"dice_{lab}"]) for r in R]) for lab in ("film", "blend", "nearest")}
        ok = ~(np.isnan(d["film"]) | np.isnan(d["blend"]) | np.isnan(d["nearest"]))
        d = {lab: v[ok] for lab, v in d.items()}
        n = int(ok.sum())
        if n < 5:
            continue
        p_b = wilcoxon(d["film"], d["blend"]).pvalue
        p_n = wilcoxon(d["film"], d["nearest"]).pvalue
        s = dict(skip=k, subset=subset, n=n,
                 dice_film=f"{d['film'].mean():.2f} ± {d['film'].std(ddof=1):.2f}",
                 dice_blend=f"{d['blend'].mean():.2f} ± {d['blend'].std(ddof=1):.2f}",
                 dice_nearest=f"{d['nearest'].mean():.2f} ± {d['nearest'].std(ddof=1):.2f}",
                 median_diff_film_minus_blend=f"{np.median(d['film'] - d['blend']):+.3f}",
                 p_film_vs_blend=f"{p_b:.3g}", p_film_vs_nearest=f"{p_n:.3g}",
                 film_better_than_blend=f"{int((d['film'] > d['blend']).sum())}/{n}")
        out.append(s)
        print(f"skip {k} {subset:5s} n={n:3d}  Dice FILM {s['dice_film']} | blend {s['dice_blend']} | "
              f"nearest {s['dice_nearest']}   FILM-blend median {s['median_diff_film_minus_blend']}, "
              f"p={s['p_film_vs_blend']} (vs nearest p={s['p_film_vs_nearest']})")
dst = os.path.join(os.path.dirname(os.path.normpath(rois[0])), "pooled_summary.csv")
with open(dst, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
print("written", dst)
