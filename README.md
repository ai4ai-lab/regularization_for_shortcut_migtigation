## Regularization for shortcut migtigation

This repository contains the implementation for the AISTATS 2025 paper "*Do Regularization Methods for Shortcut Mitigation Work As Intended?*". The project investigates the performance of different regularization methods for mitigating shortcut learning.

The experiments are conducted on
- Synthetic datasets
- Colored-MNIST
- MultiNLI
- MIMIC-IV

### Installation

1. Clone the repository:
```
git clone https://github.com/ai4ai-lab/regularization_for_shortcut_migtigation.git
```

2. Create and activate a virtual environment:
```
conda create -n regularization python=3.11
conda activate regularization
```

3. Install the required packages:
```
pip install -r requirements.txt
```

### Datasets
The project uses three different real-world datasets to evaluate shortcut mitigation methods:

1. **Colored MNIST** 

A modified version of MNIST where digits are colored either red or green, creating a potential shortcut feature. The dataset is automatically downloaded and processed when running the colored MNIST experiments.

2. **MultiNLI** 

A natural language inference dataset where the presence of negation words can create shortcuts.

1. Download the BERT base uncased model (`config.json`, and `pytorch_model.bin`) from https://huggingface.co/google-bert/bert-base-uncased/tree/main and place them in /lib/bert_base_uncased/
2. Download the MultiNLI dataset from https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz and place it in /data/multinli/

3. **MIMIC-IV** 

A healthcare dataset for length-of-stay in ICU prediction.
1. Request access to MIMIC-IV dataset
2. Download the following datasets from https://physionet.org/content/mimiciv/3.1/ and place the following files in /data/mimic_icu/icu_stays/:
    - admissions.csv.gz
    - patients.csv.gz
    - diagnoses_icd.csv.gz
    - icustays.csv.gz


### Usage

#### Data preprocessing

Each dataset has its own preprocessing script:

1. Colored MNIST:
```
python data/colored_mnist.py --device cpu
```
2. MultiNLI:
```
python data/multinli.py --train_num 1000 --test_num 200 --device cpu
```
3. MIMIC-IV:
```
python data/mimic_icu.py
```

#### Training models

The main training scripts supports different datasets. Example usage:
```
python main.py --dataset color_mnist --device cuda --reg_strengths 0.001 0.01 0.1 --num_runs 5 --num_epochs 20
```

Key parameters:

- `dataset`: Choose from `color_mnist`, `multinli`, or `mimic`
- `device`: Specify computing device (cpu or cuda)
- `reg_strengths`: List of regularization strengths to test
- `num_runs`: Number of experimental runs
- `num_epochs`: Number of training epochs

The command will generate a file called `color_mnist_experiment.pt` in the `/results` directory, which contains the trained models and their performance metrics. Use functions in `lib/plots.py` to visualize the results.

### Code Structure
- `/data/`: Dataset-specific preprocessing scripts
- `/lib/`: Core implementation of models and training utilities
- `main.py`: Main training script
- `synthetic_experiments.ipynb`: The experiments on synthetic datasets

### Citation
If you use this code or dataset, please cite the following paper:

