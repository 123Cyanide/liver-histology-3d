# 3D liver tissue geometry from serial H&E sections

Master's thesis (Heidelberg University, group of Prof. J. Hesser; supervisor Iris Kohler).
Goal: build a realistic 3D tissue geometry from serial histology for Monte Carlo
diffusion MRI simulation with disimpy.

Data: Kartasalo et al. 2018 (*Bioinformatics* 34:3013), liver, 47 H&E sections,
0.46 µm/px, 5 µm apart, with laser-drilled fiducials marked by two observers.
The data and all results are not included in this repository.

## Repository layout

| Folder | Steps | What it does |
|---|---|---|
| `registration/` | 1–4b | downsampling and fiducials (1), rigid baseline (2), CODA-style global + elastic registration in Python (3), comparison of two global variants (3b), registration applied at 1.84 or 0.46 µm/px (4), objective ROI selection (4b) |
| `interpolation/` | 5–5e | leave-out evaluation of InterpolAI (5), structure metrics / lumen Dice (5b), pooling and Wilcoxon tests (5c), merged stacks and x-z reslices (5d), montages A → generated → B (5e) |
| `finetuning/` | 6a–6c | training ROIs at full resolution (6a), fine-tuning of the FILM model used by InterpolAI (6b), paired comparison pre-trained vs fine-tuned (6c) |
| `segmentation/` | 7a–7c | VitaminP nucleus and whole-cell segmentation (7a), constrained growth of cells into tissue (7b), linking 2D cells across sections into 3D cells (7c, not yet evaluated) |

Every script documents its usage and outputs in the header docstring.

## Machines and environments

| Steps | Machine | Environment |
|---|---|---|
| 1–4b, 6a | laptop (CPU) | Python 3.10, `requirements-registration.txt` |
| 5–6c | GPU PC, RTX 4070 | InterpolAI conda env (`environment_4090.yml` of InterpolAI, TensorFlow 2.10) + `tensorflow-addons==0.18.0 gin-config`; FILM repository cloned as `frame_interpolation` |
| 7a–7c | GPU PC | conda env `vitaminp`, `requirements-segmentation.txt` |

External code: [InterpolAI](https://github.com/sjoshi17jhu/InterpolAI),
[FILM](https://github.com/google-research/frame-interpolation),
[VitaminP](https://github.com/idso-fa1-pathology/VitaminP),
CODA (Kiemen et al. 2022, re-implemented).

## Main results so far

**Registration** (target registration error between consecutive sections, observer 1, n = 184 hole pairs)

| | mean ± SD | median |
|---|---|---|
| unregistered | 726.8 ± 528.9 µm (= Kartasalo Table 2) | 585.6 µm |
| CODA-style, paper variant (Radon) | 74.8 ± 170.2 µm | 23.9 µm |
| inter-observer difference (lower bound) | 13.0 ± 23.8 µm | 6.8 µm |

**InterpolAI, leave-out test** (3 ROIs at 1.84 µm/px, clean frames, lumen Dice)

| missing sections | n | InterpolAI | linear blend | nearest | p vs nearest |
|---|---|---|---|---|---|
| 1 | 73 | 0.36 | 0.26 | 0.32 | 0.015 |
| 2 | 133 | 0.33 | 0.25 | 0.32 | 0.27 |
| 3 | 181 | 0.28 | 0.24 | 0.30 | 0.57 |

Note: `main.py --mode skip` of InterpolAI re-uses the wrong first image for every pair
after the first when skip ≥ 1; step 5 therefore loads every pair explicitly.

**Fine-tuning FILM** (test ROI not used for training, 0.46 µm/px, 1 missing section, n = 33)

| model | images | PSNR vs real | sharpness (1 = real) |
|---|---|---|---|
| pre-trained (Vimeo-90k) | sharp, smeared / abrupt switch | 17.1 dB | 1.35 |
| fine-tuned, L1 | blurry average | 19.4 dB | low |
| fine-tuned, L1 + VGG + style | natural, smooth | 16.8 dB | 1.17 |

No individual cells are visible in x-z reslices with any model.

**Segmentation** (VitaminP, sections 030/031, cell threshold 0.3): nuclei median ≈ 50 µm²;
whole cells ≈ 150 µm²; after constrained growth by 6 µm ≈ 250 µm² (≈ 18 µm),
covering 85–88 % of the tissue.
