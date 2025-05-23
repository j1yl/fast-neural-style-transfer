# Image-to-Emoji: Minimal Fast Neural Style Transfer Plan

## Project Goal
Transform real-world images into emoji-stylized visuals using a minimal, modular fast neural style transfer implementation in PyTorch.

## Requirements
- Simple, modular codebase (4 files: stylize.py, transformer.py, vgg.py, utils.py)
- Satisfies a 20-minute presentation and a 3-page report
- Demonstrates the full pipeline: data prep, model, training, inference, results

## Overview

### Day 1–2: Project Setup & Dataset Preparation
- [x] Install dependencies: torch, torchvision, Pillow, matplotlib, tqdm
- [x] Prepare dataset: Place 5–10 real-world images in data/content/, download emoji images to data/style/

### Day 3–4: Minimal Implementation

- [ ] **Preprocess Images**
  - Resize all images to 256x256, normalize using ImageNet mean/std (in utils.py)

- [ ] **Model Implementation**
  - Implement TransformerNet (transformer.py)
  - Implement Vgg16 feature extractor (vgg.py)
  - Implement utility functions (utils.py)

- [ ] **Training Script**
  - Implement minimal training loop in stylize.py
  - Use one content directory and one emoji style image for demonstration
  - Save trained model

### Day 5: Inference & Results

- [ ] **Stylization Script**
  - Implement minimal inference in stylize.py
  - Stylize a few content images with the trained model and save outputs

- [ ] **Visualization**
  - Save and display before/after images for presentation/report

### Day 6: Documentation & Presentation

- [ ] **Prepare Slides**
  - Cover all topics in PRESENTATION_OUTLINE.md
  - Include code snippets, diagrams, and results

- [ ] **Write Report**
  - 3 pages: Introduction, Method, Results, Discussion

### Day 7: Final Polish

- [ ] Review code for clarity and minimalism
- [ ] Finalize slides and report
- [ ] Submit deliverables