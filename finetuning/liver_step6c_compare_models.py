"""
Step 6c: compare the pre-trained and the fine-tuned InterpolAI model on the same
held-out ROI (paired, frame by frame).

  python liver_step6c_compare_models.py --pth D:\\thesis\\data\\roi2_full ^
      --a eval_interpolai --b eval_finetuned

Reads <pth>/<a>/per_frame.csv and structure_per_frame.csv (and the same for <b>), from
steps 5 and 5b run with --name <a> / --name <b>.
Prints, for the clean frames of every skip value: mean of each metric for a and b,
median paired difference (b - a), how often b is better, Wilcoxon p.
"""
import os, csv, argparse
import numpy as np
from scipy.stats import wilcoxon

ap = argparse.ArgumentParser()
ap.add_argument("--pth", required=True)
ap.add_argument("--a", default="eval_interpolai")
ap.add_argument("--b", default="eval_finetuned")
args = ap.parse_args()


def rd(p):
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def load(name):
    d = os.path.join(args.pth, name)
    rows = {(int(r["skip"]), int(r["A"]), int(r["target"])): r for r in rd(os.path.join(d, "per_frame.csv"))}
    sp = os.path.join(d, "structure_per_frame.csv")
    if os.path.exists(sp):
        for r in rd(sp):
            key = (int(r["skip"]), int(r["A"]), int(r["target"]))
            if key in rows:
                rows[key]["ssim74_film"], rows[key]["dice_film"] = r["ssim74_film"], r["dice_film"]
    return rows


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


A, B = load(args.a), load(args.b)
keys = sorted(set(A) & set(B))
metrics = ["psnr_film", "ssim_film", "ssim74_film", "dice_film"]
out = []
for k in sorted({key[0] for key in keys}):
    ks = [key for key in keys if key[0] == k and A[key]["clean"] == "1"]
    for m in metrics:
        x = np.array([num(A[key].get(m)) for key in ks]); y = np.array([num(B[key].get(m)) for key in ks])
        ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
        if x.size < 5:
            continue
        p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
        out.append(dict(skip=k, metric=m.replace("_film", ""), n=x.size,
                        pretrained=round(float(x.mean()), 4), finetuned=round(float(y.mean()), 4),
                        median_diff=round(float(np.median(y - x)), 4),
                        finetuned_better=f"{int((y > x).sum())}/{x.size}", p=f"{p:.3g}"))
dst = os.path.join(args.pth, f"compare_{args.a}_vs_{args.b}.csv")
with open(dst, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
print(f"clean frames, {args.a} (pre-trained) vs {args.b} (fine-tuned); higher is better for all metrics")
print(f"{'skip':>4} {'metric':>7} {'n':>4} {'pretrained':>10} {'finetuned':>10} {'median diff':>11} {'ft better':>9} {'p':>8}")
for r in out:
    print(f"{r['skip']:>4} {r['metric']:>7} {r['n']:>4} {r['pretrained']:>10} {r['finetuned']:>10} "
          f"{r['median_diff']:>11} {r['finetuned_better']:>9} {r['p']:>8}")
print("written", dst)
