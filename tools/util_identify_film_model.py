import os, argparse
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("--interpolai", default=os.path.join("interpolation", "model"))
ap.add_argument("--film", required=True, help="folder containing L1/, VGG/, Style/")
a = ap.parse_args()


def weights(path):
    m = tf.saved_model.load(path)
    w = [(tuple(v.shape), v.numpy().ravel().astype(np.float64)) for v in m.variables]
    return sorted(w, key=lambda t: (t[0], float(t[1].sum()), float(t[1][0]) if t[1].size else 0.0))


ref = weights(a.interpolai)
print(f"InterpolAI model: {len(ref)} weight tensors, {sum(w.size for _, w in ref):,} parameters")
for name in ("L1", "VGG", "Style"):
    p = os.path.join(a.film, name, "saved_model")
    if not os.path.exists(p):
        p = os.path.join(a.film, name)
    if not os.path.exists(os.path.join(p, "saved_model.pb")):
        print(f"  {name:5s}: not found at {p}")
        continue
    w = weights(p)
    if [s for s, _ in w] != [s for s, _ in ref]:
        print(f"  {name:5s}: different architecture")
        continue
    diff = max(float(np.abs(x - y).max()) for (_, x), (_, y) in zip(ref, w))
    print(f"  {name:5s}: max |weight difference| = {diff:.3g}" + ("   <-- identical" if diff < 1e-6 else ""))
