# Generative Abstract Art via Deep Learning

A research project exploring **abstract art generation** with a small reference dataset — achieving diverse, aesthetically consistent outputs from as few as **306 training images**.

📄 **[Read the Paper](https://github.com/woodyhoko/Art_Creation/blob/main/paper.pdf)**

---

## Result

Generated abstract art sample:

![Generated Art Result](https://raw.githubusercontent.com/woodyhoko/Art_Creation/main/result.png)

---

## Overview

Traditional generative art models require large-scale datasets to learn stylistic diversity. This project investigates whether a **small curated corpus of abstract art** can serve as sufficient training signal to produce novel, visually coherent outputs.

Key questions explored:
- What is the minimum viable training set size for style-consistent generation?
- How do different loss functions affect abstract aesthetic quality?
- Can perceptual similarity metrics serve as a meaningful training objective for abstract work?

---

## Repository Contents

| File | Description |
|---|---|
| `art_creation.ipynb` | Full training pipeline — data loading, model definition, training loop, generation |
| `paper.pdf` | Research paper with methodology, experiments, and analysis |
| `result.png` | Sample generated output |

---

## Stack

- Python, PyTorch
- Jupyter Notebook
- Custom GAN / generative architecture (details in paper)

---

## Usage

```bash
pip install torch torchvision jupyter
jupyter notebook art_creation.ipynb
```

