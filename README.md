# helsinki-vento-project
This repository contains code and data related to the Vento – 2nd Bootcamp (2-6 December 2023). The team Helsinki members are: Lorenzo Bergadano, Michela Androlli, Marco Tuccio, Flavio Proietti Pantosti.

The market for LTCI is growing at a steady pace (25% YoY 21-22), due to the growing number of non-self-sufficient elders and more people who experience the burden of caretaking. Modeling the risk profile of LTCI policies is difficult because people live longer and age differently, this leads to risky models for insurance companies.
Through better risk-profile and dynamic assessment for LTCI policy subscribers, Helsinki team aims to ensure that insurance companies can optimize their policies and therefore profit from the growing LTCI market

Here we train multiple Neural Networks (NNs) to learn how different vital parameters of our body may have a future impact in determining certain neurodegenerative diseases that can lead to non-self-sufficiency. This is done by analysing many datasets coming from different health data provider.

-------------------------

## Table of content
- [Cloning the repo](#cloning-the-repo)
- [Requirements](#requirements)
- [Execution](#execution)
- [Demo](#demo)
- [Directory Structure](#directory-structure)
- [Contacts](#contacts)

------------------

## Cloning the repo
To clone the repo through HTTPS or SSH, you must have installed Git on your operating system.<br>
Then you can open a new terminal and type the following command (this is the cloning through HTTPS):
```bash
    git clone https://github.com/lolloberga/helsinki-vento-project.git
```

If you don't have installed Git, you can simply download the repository by pressing <i>"Download ZIP"</i>.

## Requirements

See `requirements.txt` for the Python library requirements for running the code in this repository.
Once the repo is cloned, some Python libraries are required to properly set up your (virtual) environment.

First of all, it is strongly recommended to configure a virtual environment for this project. In order to do so, you have to launch these commands on a terminal:
```bash
    cd [...]/helsinki-vento-project
    python -m venv venv
```

Then you can install all the libraries needed via pip:
```bash
    python -m pip install -r requirements.txt
```

or via conda:
```bash
    conda create --name <env_name> --file requirements.txt
```

-----------------------
## Execution

There are two main scripts on the root folder which are a demonstrator of our model: you can run them in order to train the models and produce the results.<br>
To run the training of the model, you can run:
- `python ./stroke_main.py`: to run the training of the Stroke Detection model 
- `python ./smoke_main.py`: to run the training of the Smoke Detection model

The result of the training can be observed in real time via the Tensorboard. To activate this service, simply run the following commands on a terminal:
```bash
    cd [...]/helsinki-vento-project
    source venv/bin/activate
    tensorboard --logdir=runs
```

Tensorboard will be activated by default on your local computer at this address: `http://localhost:6006/`

------------------------

## Demo
We compute the Mean Square Error (MSE) for both the below approaches.


## Directory Structure

- [dataset](#dataset): Contains dataset files in various formats.
- [advanced](#advanced): Contains VQNN implementations for the advanced task.
- [datapreprocessing](#data-preprocessing): Contains data preprocessing code and notes.
- [model](#model): Contains VQNN model implementations.
- [model_selection](#model-selection): Hyperparameter tuning files.
- [prediction](#prediction): Classical predictions.
- [demo](#demo): demo of COMPRESS BOT.
- [plots](#plots): Contains various plots and visualizations.
- [utils](#utils): Contains utility code for plotting results.

### Dataset
Contains dataset files in various formats.

- `concrete_data.csv`: Raw dataset file.
- `dataset_with_outliers.csv`: Dataset file with outliers.
- `dataset_without_outliers.csv`: Dataset file without outliers.
- `dataset_without_outliers_without_feature.csv`: Dataset file without outliers and a specific feature.

The original dataset is public on Kaggle: https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set

### Advanced
Contains advanced VQNN implementations.

- `VQNN_basic_entangler.py`: Implementation of a basic VQNN with an entangler circuit.
- `basic_entangler_circuit.pdf`: PDF documentation for the basic entangler circuit.
- `VQNN_random_ansatz.py`: Implementation of a VQNN with a random ansatz circuit.
- `random_circuit.pdf`: PDF documentation for the random ansatz circuit.

### Data preprocessing
Contains data preprocessing code and notes.

- `datapreprocessing.ipynb`: Jupyter Notebook for data preprocessing.
- `notes.md`: Notes related to data preprocessing.

### Model
Contains VQNN model implementations.

- `VQNN_linear.ipynb`: Jupyter Notebook for linear VQNN.
- `VQNN_linear.py`: Python script for linear VQNN.
- `VQNN_nonlinear.ipynb`: Jupyter Notebook for nonlinear VQNN.
- `VQNN_nonlinear.py`: Python script for nonlinear VQNN.

### Model Selection
Contains VQNN hyperaparameters tuning and the best model.

### Plots
Contains various plots and visualizations.


### utils
Contains utility code for plotting results.

- `plot_results.ipynb`: Jupyter Notebook for plotting results.

-------------------------------------------------------------

## Contacts

| Author                | GitHub                                     | 
|-----------------------|--------------------------------------------|
| **Lorenzo Bergadano** | [lolloberga](https://github.com/lolloberga) |
| **Marco Tuccio**      | [ale100gs](https://github.com/ale100gs)    |