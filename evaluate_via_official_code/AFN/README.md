## Introduction

To evaluate the effectiveness of our suggestions, we modified and experimented with the official code provided by AFN.

The original code is taken from [here](https://github.com/WeiyuCheng/AFN-AAAI-20) and we refer to its `README.md` used for "Experimental Enviroments" and "Dataset" in this document.

## Getting Started

### 0. Experimental Enviroments

The code has been tested running under Python 3.7. The required packages are as follows:

- tensorflow == 1.10.0

### 1. Dataset

Frappe and Movielens datasets are in `./data/frappe` and `./data/movielens` respectively. If you'd like to also run experiments on Criteo and Avazu datasets, **please first run the downloading script**:

```bash
cd src
python download_criteo_and_avazu.py
```

### 2. Run

We provide scripts to run the original and modified code as follows.

**Rerun:**

```bash
bash rerun.sh
```

**Ours:**

```bash
bash run.sh
```

