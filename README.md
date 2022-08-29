# DGL Implementation of the GraphEvolveDroid Paper
This DGL example implements the GraphEvolveDroid model proposed in the paper [GraphEvolveDroid: Mitigate Model Degradation in the Scenario of Android Ecosystem Evolution](https://dl.acm.org/doi/abs/10.1145/3459637.3482118). The author's codes of implementation is [here](https://github.com/liangxun/GraphEvolveDroid).

## Dependencies
* Python 3.8.13
* PyTorch 1.10.1
* dgl 0.8.0
* scikit-learn 1.0.2
* numpy 1.21.2

## Installation
1. Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

2. Activate the new environment:
    ```bash
    conda activate pyto1.10
    ```

## Dataset
Dataset comes from paper [TESSERACT: eliminating experimental bias in malware classification across space and time](https://dl.acm.org/doi/abs/10.5555/3361338.3361389).

### data file details
* node feature matrix file
* adjacency matrix file
* labels of sample file
* dataset partition info file

## directory structure
```bash
<GraphEvovleDroid-dgl>
|-- checkpoints     # save model state dict and performance
|   |-- history_records
|   |__ reports.csv
|
|-- dataset.py      # construct customized DGL dataset class
|-- evoluNetwork.py     # construct Android evolutionary network
|-- model.py
|-- train.py
|-- README.md
|__ environment.yml
```

## How to run
1. Construct evolutionary network, run
    ```bash
    python3 evoluNetwork.py
    ```

2. Train with customized hyperparameters, run
    ```bash
    python3 train.py --num-epochs 5 --num-hidden 200 --batch-size 128 --detailed
    ```

Train on GPU, run
    ```bash
    python3 train.py --gpu 0 --num-epochs 5 --num-hidden 200 --batch-size 128 --detailed
    ```
use `nohup`, run
    ```bash
    nohup python3 train.py --gpu 0 --num-epochs 5 --num-hidden 200 --batch-size 128 --detailed >> nohup.out &
    ```

3*. Try different neighbor sampling strategy, run
    ```bash
    nohup python3 train.py --gpu 1 --num-epochs 3 --fan-out '-1' --num-layers 2 --detailed >> nohup.out &
    ```

## Q&A
TODO...