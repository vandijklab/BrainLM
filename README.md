# BrainLM

[![Preprint License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Pytorch implementation of Brain Language Model (BrainLM), aiming to achieve a general understanding of brain dynamics through self-supervised masked prediction. 
![Alt text](/figures/brainlm_overview.png)

# Quickstart
Clone this repository locally:

```
git clone https://github.com/vandijklab/BrainLM.git
```


Create an Anaconda environment from the `environment.yml` file using:

```
conda env create --file environment.yml
conda activate brainlm
```

And check the installation of major packages (Pytorch, Pytorch GPU-enabled, huggingface) by running these lines in a terminal:
```
python -c "import torch; print(torch.randn(3, 5))"
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```


# Datasets

Datasets are available on shared storage. Ask Syed or Antonio for more details.


# Training Models

To train a model on Yale HPC, see the example HPC job submission script in ```scripts/train_brainlm_mae.sh```.


# Manual Environment Creation
If the `environment.yml` file does not successfully recreate the environment for you, you can follow the below steps to install the major packages needed for this project:

1. Create and activate an anaconda environment with Python version 3.8:
```
conda create -n brainlm python=3.8
conda activate brainlm
```

2. Install Pytorch: `conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch`
3. Install latest huggingface version: `pip install git+https://github.com/huggingface/transformers`

4. Install Huggingface datasets: `conda install -c huggingface datasets`

5. Install Pandas, Seaborn, and Matplotlib: `conda install pandas seaborn`

6. Install Weights & Biases: `conda install -c conda-forge wandb`

7. Install AnnData: `pip install anndata==0.8.0`

8. Install UMAP: `pip install umap-learn`

9. Install Pytest: `conda install -c anaconda pytest`
