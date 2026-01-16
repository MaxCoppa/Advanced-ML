# Advanced-ML  
**Financial Time Series Prediction with Deep Learning**

This repository contains the code and experiments developed for the *Advanced Machine Learning* course project.  
The goal of this project is to critically evaluate state-of-the-art deep learning methods for **financial time series prediction** under realistic conditions.

The project is conducted by **Antoine Gilson**, **Marama Simoneau**, and **Maxime Coppa**.

---

## Project Overview

Financial time series prediction remains a challenging task due to the noisy, non-stationary, and weakly structured nature of market data. While recent advances in deep learning—particularly **Autoencoders** and **Transformer-based architectures**—have shown promise in sequence modeling, their actual contribution to predictive performance in financial settings remains debated.

In this work, we conduct a systematic empirical study using the **Jane Street Market Prediction** dataset (https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting). We investigate whether modern deep learning architectures provide tangible benefits over strong classical baselines, and whether representation learning through autoencoders improves downstream performance or primarily affects optimization dynamics.

---

## Our contribution

We explore three complementary modeling approaches:

### 1. Supervised Autoencoders
We train supervised autoencoders that jointly optimize:
- a **reconstruction loss** on the input features
- a **prediction loss** on the target variable  

The goal is to learn compact latent representations that retain predictive information while reducing input dimensionality.

### 2. LightGBM using Latent Features
We evaluate whether latent representations learned by supervised autoencoders improve the performance of a strong non-neural baseline:
- **LightGBM**
- Comparison between raw features and autoencoder-derived features

### 3. Transformer-based Models
We assess Transformer encoders for end-to-end forecasting:
- trained directly on raw features
- trained on autoencoder latent representations  

We deliberately constrain model capacity to test whether strong performance can be achieved without excessively complex architectures.

---

## Key Results

- Compact Transformer encoders achieve strong predictive performance, reaching **validation scores around R² ≈ 0.87**
- Supervised autoencoders effectively learn stable and meaningful latent representations
- Latent features learned by AutoEncoders :
  - improve **training stability** and **convergence speed**
  - they slightly **improve final predictive performance** do, but not significantly
- The primary benefit of deep learning in this setting appears to stem from **end-to-end optimization**, rather than representation learning alone

These results align with recent literature questioning the necessity of increasingly complex architectures for financial forecasting.

---

## Repository structure

.
├── README.md
├── requirements.txt
│
├── data/
│   ├── __init__.py
│   ├── build_data_symb1.ipynb
│   ├── data_loader_aws.py
│   └── data_preprocessing.py
│
├── models/
│   ├── autoencoders.py
│   ├── advanced_autoencoders.py
│   ├── transformers.py
│   ├── transformers_utils.py
│   ├── losses.py
│   └── train_model.py
│
├── autoencoders/
│   ├── 0_baseline_ae.ipynb
│   ├── 1_rec_weight_ae.ipynb
│   ├── 2_hyperparameters_ae.ipynb
│   └── 3_latent_representation_ae.ipynb
│
├── transformers/
│   ├── 0_baseline_tf.ipynb
│   ├── 0_baseline_tf_scores.parquet
│   ├── 1_latent_features_tf.ipynb
│   └── 1_latent_features_tf_scores.parquet
│
├── LGBM/
│   └── comparison.ipynb

## How the repository was created

The repository is structured around three layers:

1) **Data pipeline (`data/`)**  
All dataset-related utilities live here:
- loading from S3 (Onyxia)
- preprocessing / feature engineering
- building ready-to-train datasets

2) **Reusable modeling code (`models/`)**  
This folder contains the core implementations shared across experiments:
- model architectures (autoencoders, transformers)
- custom losses
- training utilities (training loops, helpers)

3) **Experiments (`autoencoders/`, `transformers/`, `LGBM/`)**  
All experiments are executed from notebooks grouped by model family:
- `autoencoders/`: supervised autoencoders experiments (baseline, reconstruction weighting, hyperparameters, latent representations)
- `transformers/`: Transformer baselines and Transformer models using latent features
- `LGBM/`: LightGBM comparisons (raw vs latent features)

Each notebook imports and reuses components from `models/` and datasets produced by the pipeline in `data/`.

---

## How to reproduce the results

### 1. Data setup (Onyxia / S3)

- First key step : Store the Jane Street dataset on your cloud Onyxia S3
- In `data/data_loader_aws.py`, make sure a user is defined:
  
  USER = "your_username"

- Update the S3 path if needed to match your bucket structure

---

### 2. Data preprocessing

Run the following steps in order:

1. Download / load the data from S3  
   - `data/data_loader_aws.py`

2. Preprocess the raw data  
   - `data/data_preprocessing.py`

3. Build the final training dataset  
   - `data/build_data_symb1.ipynb`

This produces the processed files used by all experiments.

---

### 3. Run experiments

All experiments are executed from notebooks. Depending on which architecture you want to run :

- Autoencoders:  
  `autoencoders/`

- Transformers:  
  `transformers/`

- LightGBM baseline:  
  `LGBM/comparison.ipynb`

Each notebook loads the processed data, trains the corresponding model, and evaluates performance.


