# DGL Implementation of the GraphEvolveDroid Paper
A DGL implementation for the CIKM 2021 paper below:   
GraphEvolveDroid: Mitigate Model Degradation in the Scenario of Android Ecosystem Evolution.
[[paper]](https://dl.acm.org/doi/abs/10.1145/3459637.3482118)[[author's code]](https://github.com/liangxun/GraphEvolveDroid)

## Dependencies
* Python 3.8.13
* PyTorch 1.10.1
* dgl-cuda10.2 0.8.0post2
* scikit-learn 1.0.2
* numpy 1.21.2

## Environment
1. Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

2. Activate the new environment:
    ```bash
    conda activate pyto1.10
    ```

## Dataset
Dataset comes from [TESSERACT: eliminating experimental bias in malware classification across space and time](https://dl.acm.org/doi/abs/10.5555/3361338.3361389).

## Project structure
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

2. Train on GPU, run

   ```bash
   python3 train.py --gpu 0
   ```

   or use `nohup`, run

   ```bash
   nohup python3 train.py --gpu 0 >> nohup.out &
   ```

3. Tune your hyperparameters, run

   ```bash
   python3 train.py --gpu 0 --num-epochs 5 --num-hidden 200 --batch-size 128 --detailed
   ```

4. Specify the number of sampled neighbors for each layer, run

   ```bash
   nohup python3 train.py --num-layers 2 --fan-out 5,5 --gpu 0 --detailed >> nohup.out &
   ```

5. Inductive learning on graph(**recommended**), run

   ```bash
   nohup python3 train.py --gpu 0 --detailed --inductive >> nohup.out &
   ```

6. Train one-layer GraphSAGE, run
   ```bash
   nohup python3 train.py --model sage --num-layers 1 --fan-out 10 --lr 1e-3 --weight-decay 1e-3 --num-epochs 300 --early-stop >> nohup.out &
   ```