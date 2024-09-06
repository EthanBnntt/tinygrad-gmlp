# Gated MLPs (gMLP) Implementation in Tinygrad

An implementation of gated MLPs in tinygrad, as an alternative to transformers: gMLP Tinygrad.

## What is a gMLP?

A Gated Multi-Layer Perceptron (gMLP) is a type of deep learning model that uses basic multi-layer perceptrons combined with gating mechanisms to capture spatial interactions between sequence elements, achieving comparable performance to Transformers on various language and vision tasks without relying on self-attention mechanisms.

## Installation

To get started, simply install gmlp_tinygrad using pip:

```bash
pip install gmlp_tinygrad
```

## Usage
```bash
from gmlp_tinygrad import gMLP
```

## Citations

Liu, H., Dai, Z., So, D. R., & Le, Q. V. (2021). Pay Attention to MLPs (Version 2). arXiv. https://doi.org/10.48550/ARXIV.2105.08050
