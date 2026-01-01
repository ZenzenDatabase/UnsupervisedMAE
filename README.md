# HSC-MAE  
**Hierarchical Semantic Correlation-Aware Masked Autoencoder for Unsupervised Audio–Visual Learning**

This repository contains the official implementation of **HSC-MAE**, a dual-path teacher–student framework for unsupervised audio–visual representation learning from weakly paired, label-free data.

---

## Overview

HSC-MAE addresses key challenges in unsupervised audio–visual learning, including partial observation, multi-event clips, and spurious co-occurrences.  
It enforces **hierarchical semantic correlations** at three complementary levels:

- **Sample-level (Conditional Sufficiency):** Masked autoencoding learns robust intra-modal features under partial observation.
- **Global-level (Canonical Geometry):** Deep CCA aligns audio and visual embeddings in a shared subspace.
- **Local-level (Neighborhood Semantics):** Teacher-mined soft top-k InfoNCE preserves multi-positive neighborhood structure.

The model consists of:
- a **student MAE path** trained with reconstruction and soft contrastive learning, and  
- an **EMA teacher CCA path** evaluated on clean inputs to provide stable geometry and affinity targets.

---

## Repository Structure


├── config/ # Training and evaluation configuration files
├── data_loader/ # Dataset loading and preprocessing
├── models/ # Encoders, MAE decoder, CCA modules
├── losses/ # Reconstruction, soft InfoNCE, DCCA, distillation
├── metrics/ # Retrieval metrics (mAP)
├── utils/ # EMA, masking, logging utilities
├── train.py # Training script
├── eval.py # Evaluation script
└── README.md


---

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- NumPy  
- SciPy  
- scikit-learn  
- tqdm  

Install dependencies with:

```bash
pip install -r requirements.txt

```
Data Preparation

HSC-MAE operates on pre-extracted audio and visual feature vectors rather than raw waveforms or images.

Example structure:

data/
├── AVE/
│   ├── audio_features.npy
│   └── visual_features.npy
└── VEGAS/
    ├── audio_features.npy
    └── visual_features.npy


Dataset loading can be adapted via data_loader/.

Training

Run training with:
```bash
python train.py --config config/hsc_mae.yaml
```
Training includes:

sample-level value masking (student MAE path),

EMA teacher updates (CCA path),

learnable multi-task loss weighting,

optional teacher–student distillation.

Evaluation

Evaluate cross-modal retrieval:
```bash
python eval.py --config config/hsc_mae.yaml --checkpoint path/to/checkpoint.pth
```
Metrics:

- Audio → Visual mAP

- Visual → Audio mAP

A linear CCA projection may be applied at test time to sharpen retrieval geometry.

Notes

The EMA teacher is evaluated on clean (unmasked) inputs.

Gradients are blocked from flowing into the teacher.

Teacher embeddings are used only for affinity mining and distillation targets.

Citation

If you use this code, please cite:
``` bash
@article{hscmae2026,
  title={Hierarchical Semantic Correlation-Aware Masked Autoencoder for Unsupervised Audio--Visual Learning},
  author={},
  journal={},
  year={2026}
}
```
Contact

For questions or issues, please open an issue or contact the authors.

