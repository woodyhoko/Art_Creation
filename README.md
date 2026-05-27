# StyleGAN Variants for Abstract Art Generation

**Ho Ko · Meng Yuan** — ECE, University of Waterloo · August 2021

📄 **[Read the Paper](https://github.com/woodyhoko/Art_Creation/blob/main/paper.pdf)**

---

## Abstract

> *We propose 4 variants to StyleGAN's mapping model that address its core limitations in artistic generation — reducing Wasserstein loss by ~25% on anime portraits and ~20% on abstract art while using fewer than half the parameters of the original mapping network.*

GANs have achieved remarkable results for photorealistic image generation (faces, animals), but their application to **abstract art** remains underexplored. This paper adapts **StyleGAN** for two artistic domains — anime portraits and abstract art — and identifies a fundamental limitation: the original StyleGAN applies the *same* intermediate latent vector across all resolution layers of the synthesis network, which is sub-optimal for artistic control and causes "blobbing" artifacts.

We propose four novel mapping model variants that give each resolution layer its own dedicated style control, addressing vanishing gradients, decoupled style per scale, and parameter efficiency.

---

## Key Contributions

1. **Identified StyleGAN's mapping bottleneck** — the 8-layer FC(512)×8 structure applies a single `w ∈ W` to all resolution levels, preventing resolution-specific style control and causing blob artifacts later addressed by Nvidia in StyleGAN2 via a different approach.

2. **Proposed 4 mapping model variants** — each decouples style control per resolution layer:
   - Variant 1: Direct per-resolution style injection from different mapping layers
   - Variant 2: Hierarchical shared-then-branched structure
   - Variant 3: Residual mapping network
   - Variant 4 *(best)*: Compressed architecture with fewer parameters and higher expressivity

3. **Demonstrated on two artistic datasets**:
   - 21,551 anime face images (Kaggle, scraped from getchu.com)
   - 306 abstract art images (self-collected from Wikiart — challenging small-data regime)

---

## Results

### Anime Portraits
The best variant (Variant 4, FC(128)×6) outperforms the original FC(512)×8:
- **~25% lower Wasserstein loss** (generator + discriminator)
- **<50% of the parameter count**
- Eliminates the blobbing artifact present in the original

### Abstract Art
- **~20% lower loss** vs. the original mapping structure
- The original StyleGAN mapping fails to separate color themes across resolution layers; our variant successfully learns distinct style characteristics per scale

### Sample Output

![Generated Abstract Art](https://raw.githubusercontent.com/woodyhoko/Art_Creation/main/result.png)

---

## Method

The architecture follows StyleGAN's generator/discriminator structure with our modified mapping network. Training uses the **MSG-GAN (Multi-Scale Gradients)** paradigm with 10,000 epochs per resolution layer.

```
Latent z ──► [Mapping Network Variant] ──► {w₁, w₂, ..., w₉}
                                                │
                              Each wᵢ fed to resolution layer i
                              via AdaIN (Adaptive Instance Norm)
```

By assigning a *distinct* `wᵢ` per resolution level, the model can independently control coarse style (composition, color) at low resolutions and fine style (texture, detail) at high resolutions — matching how human artists think about layers.

---

## Related Work

| Paper | Contribution |
|---|---|
| StyleGAN (Karras et al., 2019) | Baseline — style-based generator |
| StyleGAN2 (Karras et al., 2020) | Addresses blob artifacts via revised AdaIN |
| MSG-GAN | Multi-scale gradient training strategy (used in our training) |
| Auto-Encoder / VAE | Background: latent space learning |

---

## Repository Contents

| File | Description |
|---|---|
| `art_creation.ipynb` | Full pipeline — data loading, model definition, training, generation |
| `paper.pdf` | Complete research paper (9 pages) |
| `result.png` | Sample generated abstract art output |

---

## Run

```bash
pip install torch torchvision jupyter
jupyter notebook art_creation.ipynb
```

> Training from scratch requires a CUDA GPU. The notebook covers both the anime portrait and abstract art experiments described in the paper.
