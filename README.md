# Advanced-ML
This repo is a project for prediction of financial time series using state of the art deep learning methods. 

This is conducted by Antoine Gilson, Marama Simoneau and Maxime Coppa as part of Advanced ML course.


### Repository structure

├── README.md
├── requirements.txt
├── LGBM/
│   ├── comparison.ipynb
├── autoencoders/
│   ├── 0_baseline_ae.ipynb
│   ├── 1_rec_weight_ae.ipynb
│   ├── 2_hyperparameters_ae.ipynb
│   ├── 3_latent_representation_ae.ipynb
├── data/
│   ├── __init__.py
│   ├── build_data_symb1.ipynb
│   ├── data_loader_aws.py
│   └── data_preprocessing.py
├── models/
│   ├── advanced_autoencoders.py
│   └── autoencoders.py
│   └── losses.py
│   └── train_model.py
│   └── transformers.py
│   └── transformers_utils.py
├── transformers/
│   └── 0_baseline_tf.ipynb
│   └── 0_baseline_tf_scores.parquet
│   └── 1_latent_features_tf.ipynb
│   └── 1_latent_features_tf_scores.parquet

