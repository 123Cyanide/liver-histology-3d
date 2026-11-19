"""
Step 6b: fine-tune the InterpolAI model (FILM, pre-trained on Vimeo-90k videos only)
on our registered liver sections.

Training samples come from real sections only (no ground truth exists below 5 um):
  gap 2:  x0 = section i, x1 = section i+2, target = section i+1, t = 1/2
  gap 3:  x0 = i, x1 = i+3, targets i+1 (t = 1/3) and i+2 (t = 2/3)
Damaged sections are never used. Training and validation ROIs must not overlap.

Requirements (once, in the InterpolAI env on the Legion):
  pip install tensorflow-addons==0.18.0 gin-config
  cd D:\\thesis
  git clone https://github.com/google-research/frame-interpolation frame_interpolation

Run (in D:\\thesis\\InterpolAI):
  python liver_step6b_finetune_film.py --film_repo D:\\thesis\\frame_interpolation ^
      --model interpolation\\model --train D:\\thesis\\data\\train_rois ^
      --val D:\\thesis\\data\\roi2_full --out D:\\thesis\\finetuned --steps 4000

  --check_only   only load the pre-trained weights into the rebuilt network and check
                 that both give the same output (do this first)
  --vgg 0        no perceptual loss (pure L1 gives blurrier images)

Output: <out>/saved_model        best model on the validation crops (same format as
                                 interpolation\\model, usable by main.py and step 5)
        <out>/train_log.csv      step, train loss, validation L1 / PSNR
"""
import os, re, sys, csv, glob, time, types, argparse
import numpy as np
import imageio.v2 as imageio
import tensorflow as tf
for _g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_g, True)

ap = argparse.ArgumentParser()
ap.add_argument("--film_repo", required=True, help="clone of google-research/frame-interpolation, "
                "folder named frame_interpolation")
ap.add_argument("--model", default=os.path.join("interpolation", "model"))
ap.add_argument("--train", nargs="+", default=[], help="ROI folders, or folders containing ROI folders")
ap.add_argument("--val", nargs="*", default=[])
ap.add_argument("--out", default="finetuned")
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=2, help="lower to 1 if the GPU runs out of memory")
ap.add_argument("--crop", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--l1", type=float, default=1.0)
ap.add_argument("--vgg", type=float, default=0.25, help="weight of the VGG19 perceptual loss")
ap.add_argument("--style", type=float, default=40.0, help="weight of the style (Gram) loss; "
                "0.25 / 40 are the final FILM-Style weights")
ap.add_argument("--select", choices=["loss", "l1", "last"], default="loss",
                help="which checkpoint to keep: lowest validation training-loss (default), "
                     "lowest L1 (favours blur), or simply the last one")
ap.add_argument("--vgg_weights", default="imagenet", help=argparse.SUPPRESS)  # code test: None
ap.add_argument("--gaps", type=int, nargs="+", default=[2, 3])
ap.add_argument("--damaged", type=int, nargs="*", default=[5, 6, 12, 13, 14, 28])
ap.add_argument("--val_every", type=int, default=250)
ap.add_argument("--n_val", type=int, default=32, help="fixed validation crops")
ap.add_argument("--check_only", action="store_true")
ap.add_argument("--small", action="store_true", help=argparse.SUPPRESS)   # code test only
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
tf.random.set_seed(a.seed)

# ------------------------------------------------------------- rebuild FILM (time-aware)
repo = os.path.abspath(a.film_repo)
if os.path.basename(repo) != "frame_interpolation":
    sys.exit("--film_repo must be the cloned folder named 'frame_interpolation'")
sys.path.insert(0, os.path.dirname(repo))
import gin
from frame_interpolation.models.film_net import options as film_options

pkg = "frame_interpolation.models.film_net"
src = open(os.path.join(repo, "models", "film_net", "interpolator.py"), encoding="utf-8").read()
fixed = "tf.keras.layers.Lambda(lambda x: tf.ones_like(x) * 0.5)(time)"
if fixed not in src:
    sys.exit("unexpected FILM version: time handling not found in interpolator.py")
# the released FILM code fixes t = 0.5; the InterpolAI model uses the given t (it produces
# different images for different t), so we use t here as well
src = src.replace(fixed, "tf.keras.layers.Lambda(lambda x: tf.identity(x))(time)")
interp = types.ModuleType(pkg + ".interpolator_t")
interp.__package__ = pkg
exec(compile(src, "interpolator_t.py", "exec"), interp.__dict__)

cfg = ("film_net.pyramid_levels = 7\nfilm_net.fusion_pyramid_levels = 5\n"
       "film_net.specialized_levels = 3\nfilm_net.sub_levels = 4\n"
       "film_net.flow_convs = [3, 3, 3, 3]\n")
cfg += ("film_net.flow_filters = [8, 16, 32, 64]\nfilm_net.filters = 8\n" if a.small else
        "film_net.flow_filters = [32, 64, 128, 256]\nfilm_net.filters = 64\n")
gin.parse_config(cfg)


def build():
    x0 = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32, name="x0")
    x1 = tf.keras.Input(shape=(None, None, 3), dtype=tf.float32, name="x1")
    t = tf.keras.Input(shape=(1,), dtype=tf.float32, name="time")
    return interp.create_model(x0, x1, t, film_options.Options())


model = build()
pre = tf.saved_model.load(a.model)


def clean(name):
    return re.sub(r"_\d+(?=/|:)", "", name.split(":")[0])   # drop Keras uniquifying suffixes


pre_vars = {}
for v in pre.variables:
    pre_vars.setdefault((clean(v.name), tuple(v.shape)), []).append(v)
assigned, missing = 0, []
for v in model.variables:
    key = (clean(v.name), tuple(v.shape))
    if pre_vars.get(key):
        v.assign(pre_vars[key].pop(0)); assigned += 1
    else:
        missing.append(v.name)
if missing:
    # fall back to matching by order and shape (same architecture, same creation order)
    pv = list(pre.variables)
    if len(pv) == len(model.variables) and all(tuple(p.shape) == tuple(m.shape)
                                                for p, m in zip(pv, model.variables)):
        for p, m in zip(pv, model.variables):
            m.assign(p)
        assigned, missing = len(pv), []
        print("weights matched by order")
if missing:
    sys.exit(f"could not match {len(missing)} of {len(model.variables)} weights, e.g. {missing[:3]}")
print(f"loaded {assigned} pre-trained weight tensors into the rebuilt network")


def run(m, x0, x1, t, training=False):
    out = m({"x0": x0, "x1": x1, "time": t}, training=training) if m is model else \
          m({"x0": x0, "x1": x1, "time": t})
    return out["image"]


# ------------------------------------------------------------- data
def roi_dirs(paths):
    out = []
    for p in paths:
        if glob.glob(os.path.join(p, "[0-9][0-9][0-9].png")):
            out.append(p)
        else:
            out += sorted(d for d in glob.glob(os.path.join(p, "*"))
                          if glob.glob(os.path.join(d, "[0-9][0-9][0-9].png")))
    return out


def load_roi(d):
    ims = {}
    for f in glob.glob(os.path.join(d, "[0-9][0-9][0-9].png")):
        ims[int(os.path.basename(f)[:3])] = imageio.imread(f)[..., :3]
    return ims


def samples(ims):
    bad = set(a.damaged)
    S = []
    for g in a.gaps:
        for i in ims:
            if all((i + j) in ims and (i + j) not in bad for j in range(g + 1)):
                S += [(i, i + j, i + g, j / g) for j in range(1, g)]
    return S


def batch_from(rois, S, n, crop, fixed_rng=None):
    r = fixed_rng or rng
    X0, X1, Y, T = [], [], [], []
    for _ in range(n):
        k = r.integers(len(S))
        ri, (i0, it, i1, t) = S[k]
        A, M, B = rois[ri][i0], rois[ri][it], rois[ri][i1]
        h, w = A.shape[:2]
        y, x = r.integers(0, h - crop + 1), r.integers(0, w - crop + 1)
        trip = [im[y:y + crop, x:x + crop] for im in (A, M, B)]
        if fixed_rng is None:                       # augmentation for training only
            rot = r.integers(4)
            trip = [np.rot90(im, rot) for im in trip]
            if r.random() < 0.5:
                trip = [im[:, ::-1] for im in trip]
            if r.random() < 0.5:                    # reverse direction
                trip, t = trip[::-1], 1 - t
        X0.append(trip[0]); Y.append(trip[1]); X1.append(trip[2]); T.append([t])
    f = lambda L: tf.constant(np.stack(L).astype(np.float32) / 255.0)
    return f(X0), f(X1), f(Y), tf.constant(np.array(T, np.float32))


val_dirs = roi_dirs(a.val)
train_dirs = [d for d in roi_dirs(a.train)
              if os.path.abspath(d) not in {os.path.abspath(v) for v in val_dirs}]
val_rois = [load_roi(d) for d in val_dirs]
VS = [(ri, s) for ri, R in enumerate(val_rois) for s in samples(R)]
if VS:
    vb = batch_from(val_rois, VS, a.n_val, a.crop, np.random.default_rng(123))

# ------------------------------------------------------------- check
probe = tuple(v[:1] for v in vb) if VS else (tf.random.uniform((1, a.crop, a.crop, 3)),) * 3 + (tf.fill((1, 1), 0.5),)
for t in (0.5, 0.25):
    tt = tf.fill((probe[0].shape[0], 1), t)
    d = float(tf.reduce_max(tf.abs(run(model, probe[0], probe[1], tt) - run(pre, probe[0], probe[1], tt))))
    print(f"check t = {t}: max |rebuilt - pre-trained| = {d:.2e}")
    if d > 1e-3:
        sys.exit("rebuilt network does not reproduce the pre-trained model - stop")
if a.check_only:
    sys.exit("check passed")

train_rois = [load_roi(d) for d in train_dirs]
TS = [(ri, s) for ri, R in enumerate(train_rois) for s in samples(R)]
print(f"train: {len(train_dirs)} ROIs, {len(TS)} samples | val: {len(val_dirs)} ROIs, {len(VS)} samples")
if not TS:
    sys.exit("no training samples - check --train")

# ------------------------------------------------------------- losses (as FILM-Style)
# FILM vgg19_loss.py: layers conv1_2 ... conv5_2, layer weights below, images in 0..255,
# perceptual loss / 255, style loss on Gram matrices of features / 255.
LW = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
vgg_feat = None
if a.vgg > 0 or a.style > 0:
    base = tf.keras.applications.VGG19(include_top=False,
                                       weights=None if a.vgg_weights == "None" else a.vgg_weights)
    names = [f"block{i}_conv2" for i in range(1, 6)]
    vgg_feat = tf.keras.Model(base.input, [base.get_layer(n).output for n in names])
    vgg_feat.trainable = False


def gram(f):
    b, h, w, c = tf.unstack(tf.shape(f))
    x = tf.reshape(f, (b, h * w, c))
    return tf.matmul(x, x, transpose_a=True) / tf.cast(h * w, tf.float32)


def loss_parts(pred, y):
    parts = {"l1": tf.reduce_mean(tf.abs(pred - y))}
    if vgg_feat is not None:
        fp = vgg_feat(tf.keras.applications.vgg19.preprocess_input(pred * 255.0))
        fy = vgg_feat(tf.keras.applications.vgg19.preprocess_input(y * 255.0))
        parts["vgg"] = tf.add_n([w * tf.reduce_mean(tf.abs(p - q)) for w, p, q in zip(LW, fp, fy)]) / 255.0
        parts["style"] = tf.add_n([w * tf.reduce_mean(tf.square(gram(p / 255.0) - gram(q / 255.0)))
                                   for w, p, q in zip(LW, fp, fy)])
    return parts


def total(parts):
    t = a.l1 * parts["l1"]
    if "vgg" in parts:
        t += a.vgg * parts["vgg"] + a.style * parts["style"]
    return t


opt = tf.keras.optimizers.Adam(a.lr)


@tf.function
def step(x0, x1, y, t):
    with tf.GradientTape() as tape:
        loss = total(loss_parts(run(model, x0, x1, t, training=True), y))
    g = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(g, model.trainable_variables))
    return loss


LAP = tf.constant([[0, 1, 0], [1, -4, 1], [0, 1, 0]], tf.float32)[:, :, None, None]


def sharp(img):
    """variance of the Laplacian of the grey image: drops when an image gets blurry"""
    g = tf.reduce_mean(img, axis=-1, keepdims=True)
    return tf.math.reduce_variance(tf.nn.conv2d(g, LAP, 1, "VALID"), axis=[1, 2, 3])


def evaluate(m):
    r = {"l1": [], "psnr": [], "loss": [], "sharpness": []}
    for k in range(vb[0].shape[0]):
        sl = slice(k, k + 1)
        p = tf.clip_by_value(run(m, vb[0][sl], vb[1][sl], vb[3][sl]), 0, 1)
        parts = loss_parts(p, vb[2][sl])
        r["l1"].append(float(parts["l1"]))
        r["loss"].append(float(total(parts)))
        r["psnr"].append(float(tf.reduce_mean(tf.image.psnr(p, vb[2][sl], 1.0))))
        r["sharpness"].append(float(sharp(p)[0] / (sharp(vb[2][sl])[0] + 1e-12)))
    return {k: float(np.mean(v)) for k, v in r.items()}


def fmt(e):
    return (f"L1 {e['l1']:.4f}, PSNR {e['psnr']:.2f} dB, loss {e['loss']:.4f}, "
            f"sharpness {e['sharpness']:.2f} (1 = as sharp as the real section)")


os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
w = csv.writer(log)
w.writerow(["step", "train_loss", "val_l1", "val_psnr", "val_loss", "val_sharpness"])
key = {"loss": "loss", "l1": "l1", "last": None}[a.select]
best = None
if VS:
    e0 = evaluate(model)
    print("step 0 (pre-trained):", fmt(e0))
    w.writerow([0, "", e0["l1"], e0["psnr"], e0["loss"], e0["sharpness"]])
    best = e0[key] if key else None
t0, run_loss = time.time(), []
for it in range(1, a.steps + 1):
    run_loss.append(float(step(*batch_from(train_rois, TS, a.batch, a.crop))))
    if it % 50 == 0:
        print(f"step {it}: train loss {np.mean(run_loss[-50:]):.4f}  [{time.time() - t0:.0f} s]")
    if it % a.val_every == 0 or it == a.steps:
        row = [it, f"{np.mean(run_loss[-a.val_every:]):.5f}"]
        e = evaluate(model) if VS else None
        if e:
            row += [e["l1"], e["psnr"], e["loss"], e["sharpness"]]
            print("  validation:", fmt(e))
        w.writerow(row); log.flush()
        if key is None or e is None or e[key] < best:
            if key and e:
                best = e[key]
            model.save(os.path.join(a.out, "saved_model"), include_optimizer=False)
            print(f"  saved {os.path.join(a.out, 'saved_model')} (step {it})")
log.close()
model.save(os.path.join(a.out, "saved_model_last"), include_optimizer=False)
print("last step saved to", os.path.join(a.out, "saved_model_last"))
if not os.path.exists(os.path.join(a.out, "saved_model")):
    print("validation never improved on the pre-trained model - no model saved")
print("done")
