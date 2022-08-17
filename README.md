

![logo](./document/figures/logo.png)

--------------------------------------------------------------------------------

[![License](https://img.shields.io/badge/License-Apache2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.6|3.7-success)](./README.md)

This is the repository of our empirical study on CTR prediction. With this code and the corresponding dataset, you can reproduce the results of our experiments easily.

## Models

We have implemented the following CTR models: 

| Publication |                         Model                          | Paper                                                        |
| :---------: | :----------------------------------------------------: | :----------------------------------------------------------- |
|   WWW'07    |          [LR](./libctr/pytorch/models/LR.py)           | [Predicting Clicks: Estimating the Click-Through Rate for New Ads](https://dl.acm.org/citation.cfm?id=1242643) |
|   ICDM'10   |          [FM](./libctr/pytorch/models/FM.py)           | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
|   DLRS'16   |   [Wide&Deep](./libctr/pytorch/models/Wide&Deep.py)    | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) |
|   NIPS'16   |        [HOFM](./libctr/pytorch/models/HOFM.py)         | [Higher-Order Factorization Machines](https://papers.nips.cc/paper/6144-higher-order-factorization-machines.pdf) |
|  IJCAI'17   |      [DeepFM](./libctr/pytorch/models/DeepFM.py)       | [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247) |
|  SIGIR'17   |         [NFM](./libctr/pytorch/models/NFM.py)          | [Neural Factorization Machines for Sparse Predictive Analytics](https://dl.acm.org/citation.cfm?id=3080777) |
|  ADKDD'17   |     [CrossNet/DCN](./libctr/pytorch/models/DCN.py)     | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) |
|   KDD'18    |   [CIN/xDeepFM](./libctr/pytorch/models/xDeepFM.py)    | [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf) |
|   CIKM'19   | [AutoInt/AutoInt+](./libctr/pytorch/models/AutoInt.py) | [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) |
|   AAAI'20   |       [AFN/AFN+](./libctr/pytorch/models/AFN.py)       | [Adaptive Factorization Network: Learning Adaptive-Order Feature Interactions](https://ojs.aaai.org/index.php/AAAI/article/view/5768) |
|   WWW'21    |      [DCNv2](./libctr/pytorch/models/AutoInt.py)       | [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://dl.acm.org/doi/10.1145/3442381.3450078) |

You can find the model implementation(`Model`) or the original paper by clicking on these links.

## Datasets

| Dataset   | Dataset Description                                          | Dowload                                                    |
| :-------- | :----------------------------------------------------------- | ---------------------------------------------------------- |
| Criteo    | This is a popular industry benchmarking dataset for CTR prediction, which contains 13 numerical feature fields and 26 categorical feature fields. | [⏬](https://zenodo.org/record/5700987/files/Criteo_x1.zip) |
| Avazu     | This dataset contains users’ click records on mobile advertisements. It has 22 feature fields including user features and advertisement attributes. | [⏬](https://zenodo.org/record/5700987/files/Avazu_x1.zip)  |
| Frappe    | This dataset contains app usage logs from users under different contexts (e.g., daytime, location). We converted each log (user ID, app ID, context features) to a feature vector as input. The target value indicates whether the user has used the app under the context. | [⏬](./data/Frappe/)                                        |
| Movielens | This dataset consists of users’ tagging records on movies. We focus on personalized tag recommendation by converting each tagging record (user ID, movieID, tag) to a feature vector as input. The target value denotes whether the user has assigned a particular tag to the movie. | [⏬](./data/Movielens)                                      |

In our study, we used the above four dataset including Criteo, Avazu, Frappe and Movielens. And the processed datasets are already available on Zenodo and Github for downloading by clicking ⏬. 

All data splits are fully consistent with [AFN](https://ojs.aaai.org/index.php/AAAI/article/view/5768).

## Getting Started

 ### 0. Experimental Enviroments

Our code works with the python 3.6/3.7,  pytorch, cudatoolkit  and cud. In addition, our code requires the following dependency packages:

```
- pyyaml >=5.1
- scikit-learn
- pandas
- numpy
- h5py
- tqdm
```

you can install these packages with pip:

```bash
pip install --ignore-installed pyyaml==5.1
pip install scikit-learn pandas numpy h5py tqdm
```

What's more, we tested our code with the python 3.7, pytorch 1.10.1, cudatoolkit 11.3 and cudnn 8.2.1.

 ### 1. Download datasets.

Here we provide the Frappe and Movielens datasets in `data/frappe` and `data/movielens`. The other two processed datasets (Criteo and Avazu) are already available for download, refer [here](./README.md#Datasets).

### 2. Setup model config file

The config files for all models are been placed in `config/{model_name}` including at least `model_config.yaml` and `dataset_config.yaml`. 

- `dataset_config.yaml` usually does not need to be modified unless the path where the dataset is stored changes. 

- `model_config.yaml` needs to be simply understood, or may need to be changed to suit your needs. We show the parts of the DCN's model config(`config/DCN/model_config.yaml`).

  ```yaml
  Base:
      model_root: './checkpoints/'
      workers: 8
      verbose: 1
      patience: 2
      pickle_feature_encoder: True
      use_hdf5: True
      save_best_only: True
      every_x_epochs: 1
      debug: False
      version: 'pytorch'
  
  DCN_Frappe:
      batch_norm: False
      layer_norm: False
      batch_size: 4096
      crossing_layers: 3
      dataset_id: Frappe
      dnn_activations: relu
      dnn_hidden_units: [400, 400, 400]
      embedding_dim: 10
      embedding_regularizer: 0.1
      epochs: 100
      learning_rate: 0.001
      loss: binary_crossentropy
      metrics: [logloss, AUC]
      model: DCN
      monitor: {AUC: 1, logloss: -1}
      monitor_mode: max
      net_dropout: 0.1
      net_regularizer: 0
      optimizer: adam
      seed: 2021
      shuffle: True
      task: binary_classification
  ```

  where `DCN_Frappe` is the `expid` of the experiment.

### 3. Run

After the previous steps are completed, we start training and evaluating a CTR model as follows:

```bash
python -u main.py --config config/{Model} --expid {expid} --gpu {gpu_id}
```

For example，we use to train DCN on Frappe:

```bash
# gpu:
python -u main.py --config config/DCN --expid DCN_Frappe --gpu 0
# cpu:
python -u main.py --config config/DCN --expid DCN_Frappe --gpu -1
```

### 4. Impact of Different Factor

In our study, we explored the effect of different factors on the CTR model. 

To facilitate running multiple sets of experiments at the same time, we provide scripts for tuning parameters: 

```bash
python -u run_param_tuner.py --config config/{Model}/{tuner_config.yaml} --gpu {gpu_ids}
```

`tuner_config.yaml` refers to the configuration file, which sets multiple parameters for the experiment, such as `config/DCN/order_frappe_dcn.yaml` :

```yaml
base_config: ./config/DCN
base_expid: DCN_Frappe
dataset_id: Frappe

dataset_config:
    Frappe:
        data_format: h5
        data_root: './data/'
        train_data: ./data/Frappe/train.h5
        valid_data: ./data/Frappe/valid.h5
        test_data: ./data/Frappe/test.h5

tuner_space:
    model_root: './checkpoints/'
    crossing_layers: [1,2,3,4,5,6,7,8,9]
```

`gpu_ids` is used to set the gpu number, you can set more than one gpu such as `--gpu 0 1 2 3`.

Specifically, here is an example:

```bash
python -u run_param_tuner.py --config config/DCN/order_frappe_dcn.yaml --gpu 0 1 2 3
```

Note: If you have not run the script with the corresponding dataset before, you need to refer to [here](./README.md#3-run) to run the script once first.

For convenience, we have compiled the bash scripts involved in the stud:

```bash
# Weights Initializer
bash run_explore_initializer.sh
# Regularization
bash run_explore_regularization.sh
# Order
bash run_explore_order.sh
# Normalization
bash run_explore_normalization.sh
# The number of DNN layers
bash run_explore_DNN_layers.sh
# Early Stop
bash run_explore_earlystop.sh
# Batch Size
bash run_explore_batchsize.sh
# Learning Rate
bash run_explore_lr.sh
```

### 5. Evaluate via Official Code

To evaluate the suggestions proposed in our study, we modified the official code of other models, as detailed in [here](./evaluate_via_official_code/AFN/).

## Other

To ensure reproducibility, we will release the code and supporting documentation as soon as our work is accepted. 	
