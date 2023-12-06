# Data Augmentation of PPG Signals Through Guided Diffusion

This repository contains PyTorch implemenations of Data Augmentation of PPG Signals Through Guided Diffusion by Python.

<img src="./images/flowchart.png" width="500px"></img>

## Introduction
This is a machine learning code that uses coronavirus cluster data to predict how many people each cluster will infect and how long the cluster will last. 
For the layer that embeds each information, there are two layers: a layer that processes cluster information (e.g., severity index etc.) and a layer that processes patient information (e.g., age, address, etc.).
For the Transformer model, we used a sequence of dates for each cluster and performed a regression using the correlation of each sequence.

### Project Tree
```
├── data
│   ├── data_cut_0.csv
│   ├── data_cut_1.csv
│   ├── data_cut_2.csv
│   ├── data_cut_3.csv
│   ├── data_cut_4.csv
│   ├── data_cut_5.csv
│   ├── data_final_mod.csv
│   ├── data_mod.ipynb
│   └── data_task.csv
├── sh
│   ├── linear_raw.sh
│   ├── linear.sh
│   ├── mlp_raw.sh
│   ├── mlp.sh
│   ├── ridge_raw.sh
│   ├── ridge.sh
│   └── transformer.sh
├── main.py
├── ml_algorithm.py
├── models.py
├── utils.py
└── README.md
```

For our experiments, we divided the dataset according to how observable the dynamics were. For example, if we observed a cluster until day 2 and predicted the duration of the cluster and additional patients for the remaining days, we would have ``` ./data/data_cut_2.csv ```. This generated a dataset of 5 days, which we combined into ``` data_cut_0.csv ``` and used in the experiment. The data preprocessing method is documented in ```data_mod.ipynb```.

Also, for hyperparameter sweeping, we used files located in ```./sh ```.

## Implementation

We used the following Python packages for core development. We tested on `Python 3.10.10`.
```
pytorch                   2.0.0
pandas                    2.0.0
numpy                     1.23.5
scikit-learn              1.2.2
scipy                     1.10.1
```

### Arg Parser

The script `main.py` allows to train and evaluate all the baselines we consider.

To train proposed methods use this:
```
main.py --epochs=<EPOCHS>                      \
        --model=<MODEL>                        \
        --num_features=<NUM_FEATURES>          \
        --hidden_dim=<HIDEN_DIM>               \
        --num_layers=<NUM_LAYERS>              \
        --num_heads=<NUM_HEADS>                \
        [--disable_embedding]                  \
        --wd=<WD>                              \
        --lr_init=<LR_INIT>                    \
        --drop_out=<DROP_OUT>                  \
        --lamb=<LAMB>                          \
        --scheduler=<SCHEDULAR>                \
        --t_max=<T_MAX>                        \
        [--ignore_wandb]
```
Parameters:
* ```EPOCHS``` &mdash; number of training epochs (default: 300)
* ```MODEL``` &mdash; model name (default: transformer) :
    - transformer
    - linear
    - ridge
    - mlp
    - svr
    - rfr
* ```NUM_FEATURES``` &mdash; feature size (default : 128)
* ```HIDEN_DIM``` &mdash; DL model hidden size (default : 64)
* ```NUM_LAYERS``` &mdash; DL model layer num (default : 3)
* ```NUM_HEADS``` &mdash; Transformer model head num (default : 2)
* ```--disable_embedding``` &mdash; Disable embedding to use raw data 
* ```WD``` &mdash; weight decay (default: 5e-4)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.005)
* ```DROP_OUT``` &mdash; dropout rate (default: 0.0)
* ```LAMB``` &mdash; Penalty term for Ridge Regression (Default : 0)
* ```scheduler``` &mdash; schedular (default: constant) :
    - constant
    - cos_anneal
* ```t_max``` &mdash; T_max for Cosine Annealing Learning Rate Scheduler (Default : 300)
* ```--ignore_wandb``` &mdash; Ignore WandB to do not save results

----

### Train Models

To train model for example, use this:

```
# linear
python3 main.py --model=linear --optim=adam --lr_init=1e-4 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 

# mlp
python3 main.py --model=mlp --hidden_dim=128 --optim=adam --lr_init=1e-4 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.1 --num_layers=3

# transformer
python3 main.py --model=transformer --hidden_dim=128 --optim=adam --lr_init=1e-4 --wd=1e-3 --epochs=200 --scheduler=cos_anneal --t_max=200 --drop_out=0.1 --num_layers=2 --num_heads=2

```

If you want to sweep model to search best hyperparameter, you can use this:

```
# linear
bash sh/linear.sh

# mlp
bash sh/mlp.sh 

# transformer
bash sh/transformer.sh

```

It should be modified for appropriate parameters for personal sweeping

### Evaluate Models

To test model, use this:
```
# linear
python3 main.py --model=linear --hidden_dim=128 --eval_model=<MODEL_PATH>

# mlp
python3 main.py --model=mlp --hidden_dim=128 --num_layers=3 --eval_model=<MODEL_PATH>

# transformer
python3 main.py --model=transformer --hidden_dim=128 --num_layers=2 --num_heads=2 --eval_model=<MODEL_PATH>
```

