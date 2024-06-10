# MLE_Lab1

# ML Training Pipeline

This project demonstrates a basic ML training pipeline for CIFAR-10 dataset.

## Project Structure

- `config/`: Configuration files.
- `data/`: Scripts for data download and processing.
- `models/`: Model architectures.
- `notebooks/`: Jupyter notebooks for data analysis.
- `scripts/`: Scripts for training and evaluation.
- `utils/`: Utility scripts such as logger.
- `pyproject.toml`: Project dependencies.

## Getting Started

### Installation

# 1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-training-pipeline.git
   cd ml-training-pipeline
# 2. Install dependencies using poetry:

   poetry install

## Usage

# 1. Download data:

   python datadownload.py

# 2. Train the model

   poetry run python scripts_train.py

# 3. Evaluate the model:

   poetry run python scripts_evaluate.py
