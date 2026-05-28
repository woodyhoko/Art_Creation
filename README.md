# StyleGAN Variants for Abstract Art Generation

*Four modified mapping network architectures that decouple per-resolution style control, reducing Wasserstein loss by ~25% on anime portraits and ~20% on abstract art at under half the parameter count.*

**Ho Ko · Meng Yuan** — ECE, University of Waterloo · August 2021

📄 **[Read the Paper](https://github.com/woodyhoko/Art_Creation/blob/main/paper.pdf)** | **[▶ Flow Field Demo](demo.html)**

---

## 1. Abstract

> *We propose 4 variants to StyleGAN's mapping network that address its core limitations in artistic generation — reducing Wasserstein loss by ~25% on anime portraits and ~20% on abstract art while using fewer than half the parameters of the original mapping network.*

GANs have achieved photorealistic results for natural image generation, but application to **abstract art** remains challenging due to the sparse, high-variance nature of artistic datasets and the difficulty of defining perceptual similarity in non-photographic domains. This paper adapts StyleGAN to two artistic datasets and identifies a fundamental architectural bottleneck.

---

## 2. Background: StyleGAN and the mapping network

StyleGAN (Karras et al. 2019) separates the generator into two components:

1. **Mapping network** *f: Z → W* — an 8-layer MLP that transforms a Gaussian latent code **z** ∈ ℝ^{512} into an intermediate latent code **w** ∈ ℝ^{512}. The disentangled *W* space was shown to produce more controllable image generation than sampling directly in *Z*.

2. **Synthesis network** *g: W → X* — a progressive upsampling network (4²→1024²) where each resolution layer *l* receives **w** via **Adaptive Instance Normalization (AdaIN)**:

```
AdaIN(xᵢ, w) = ys(w) · (xᵢ − μ(xᵢ)) / σ(xᵢ)  +  yb(w)
```

where *ys*, *yb* are learned affine transformations of **w**.

**The identified bottleneck:** the original StyleGAN applies the *same* **w** vector to all resolution layers. The low-resolution layers (controlling composition and color) and high-resolution layers (controlling texture and detail) share a single style code, preventing independent control at different scales. This causes **blobbing artifacts** — undesired blotches at intermediate resolutions — particularly visible on non-photographic targets like abstract art.

---

## 3. Proposed mapping network variants

All four variants generate a *set* {**w₁**, **w₂**, …, **w₉**} — one dedicated style vector per resolution layer — rather than a single shared **w**.

### Variant 1 — Direct per-resolution injection
The mapping network branches into 9 parallel heads, each outputting a separate **wᵢ**. Full independence across resolutions; highest parameter count of the four variants.

### Variant 2 — Hierarchical shared-then-branched
A shared trunk of 4 FC layers extracts a common representation, followed by 4 parallel branches (each 2 layers) producing {**wᵢ**}. Balances parameter sharing and resolution independence.

### Variant 3 — Residual mapping
Each **wᵢ** = **w_base** + **Δwᵢ**, where **w_base** is produced by the full original mapping network and **Δwᵢ** are small learned residuals per layer. The residuals allow fine-grained per-resolution adjustment without discarding the globally disentangled **w_base**.

### Variant 4 — Compressed architecture *(best)*
A single FC(128)×6 network (vs. the original FC(512)×8) produces the base representation, then 9 lightweight heads (1 layer each) extract per-resolution styles. The compressed bottleneck acts as an information regularizer — it forces the network to learn a compact, disentangled representation rather than memorizing texture.

```
Latent z ──► FC(128)×6 ──► 9 × FC(128→512) ──► {w₁, w₂, ..., w₉}
                                                        │
                                       Each wᵢ → AdaIN at resolution layer i
```

---

## 4. Training

**Framework:** MSG-GAN (Multi-Scale Gradients GAN, Karnewar & Wang 2020) — gradient signals from all resolution levels are fed to the discriminator simultaneously, stabilizing training on small datasets.

**Training regime:** 10,000 epochs per resolution layer (progressive growing), Adam optimizer (lr = 0.001, β₁ = 0.0, β₂ = 0.99), batch size 4–8.

**Loss:** Wasserstein distance with gradient penalty (WGAN-GP, Gulrajani et al. 2017):

```
L = E[D(G(z))] − E[D(x)] + λ · E[(||∇D(x̂)||₂ − 1)²]
```

Wasserstein distance provides more stable gradients than the original GAN binary cross-entropy, which is critical for small-dataset artistic generation.

---

## 5. Datasets

| Dataset | Size | Source | Challenge |
|---|---|---|---|
| Anime portraits | 21,551 images | Kaggle (scraped from getchu.com) | High within-class variance in style |
| Abstract art | 306 images | Self-collected from Wikiart | Extremely small dataset; high variance |

The abstract art setting is particularly challenging: 306 training images is orders of magnitude smaller than typical GAN training sets (10K–1M). MSG-GAN's multi-resolution gradients provide additional training signal that partially compensates.

---

## 6. Results

### Anime portraits
| Model | Wasserstein loss | Parameters |
|---|---|---|
| Original FC(512)×8 | baseline | 100% |
| Variant 4 FC(128)×6 | **−25%** | **< 50%** |

Variant 4 eliminates the blobbing artifact visible at 64² and 128² resolution in the original.

### Abstract art
- **−20% lower Wasserstein loss** vs. original mapping structure
- The original mapping fails to separate color themes across resolution layers; Variant 4 successfully learns distinct style per scale

### Sample output

![Generated Abstract Art](https://raw.githubusercontent.com/woodyhoko/Art_Creation/main/result.png)

---

## 7. Relation to StyleGAN2

Nvidia's StyleGAN2 (Karras et al. 2020) addresses the same blobbing artifact via **weight demodulation** (replacing AdaIN with a normalization of the convolution weights themselves). Our approach is complementary — decoupling the *input style codes* per layer rather than changing the *normalization mechanism*. The two approaches could be combined.

---

## 8. Repository contents

| File | Description |
|---|---|
| `art_creation.ipynb` | Full pipeline — data loading, model definition, MSG-GAN training loop, generation |
| `paper.pdf` | Complete research paper (9 pages, IEEE format) |
| `result.png` | Sample generated abstract art |
| `data3/` | Processed dataset directory |
| `demo.html` | Generative flow-field art demo (Perlin noise, browser) |

---

## 9. Run

```bash
pip install torch torchvision jupyter
jupyter notebook art_creation.ipynb
```

> Training from scratch requires a CUDA GPU. The notebook covers both datasets described in the paper.

---

## 10. References

1. T. Karras et al. "A Style-Based Generator Architecture for Generative Adversarial Networks." *CVPR 2019*.
2. T. Karras et al. "Analyzing and Improving the Image Quality of StyleGAN." *CVPR 2020*.
3. A. Karnewar and O. Wang. "MSG-GAN: Multi-Scale Gradients for Generative Adversarial Networks." *CVPR 2020*.
4. I. Gulrajani et al. "Improved Training of Wasserstein GANs." *NeurIPS 2017*.
5. I. J. Goodfellow et al. "Generative Adversarial Nets." *NeurIPS 2014*.
