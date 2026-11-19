import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = sys.argv[1] if len(sys.argv) > 1 else "liver_out"
T, S = "1.5", "1.5"
res = {}
for mode in ("paper", "refined"):
    f = os.path.join(OUT, f"coda_python_{mode}_tile{T}_spacing{S}", "tre_per_pair.csv")
    old = os.path.join(OUT, f"coda_python_tile{T}_spacing{S}", "tre_per_pair.csv")
    if mode == "refined" and not os.path.exists(f) and os.path.exists(old):
        f = old                                   # earlier run of step 3 (= refined mode)
        print("refined: using the earlier step-3 run in", os.path.dirname(f))
    if not os.path.exists(f):
        sys.exit(f"missing {f} - run step 3 with MODE={mode} first")
    res[mode] = pd.read_csv(f, index_col=0)

rows = []
for col in ("CODA-style global (rigid)", "CODA-style global + elastic"):
    p, r = res["paper"][col], res["refined"][col]
    common = p.index.intersection(r.index)
    p, r = p[common].to_numpy(), r[common].to_numpy()
    rows.append(dict(step=col, n_pairs=len(common),
                     paper_mean=p.mean(), paper_median=np.median(p),
                     refined_mean=r.mean(), refined_median=np.median(r),
                     median_diff_refined_minus_paper=np.median(r - p),
                     refined_better=f"{int((r < p).sum())}/{len(common)}",
                     wilcoxon_p=wilcoxon(r, p).pvalue))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "step3b_compare_modes.csv"), index=False)
pd.set_option("display.width", 200)
print("TRE per section pair, observer 1 [um]  (paper = Radon angle, refined = sweep + ECC)")
print(df.round(3).to_string(index=False))
