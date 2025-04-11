# FederatedLearning

This repository contains code for emulating federated learning scenarios and comparing them with centralized learning. It includes implementations for different federated averaging strategies and batch-wise gradient transmission, along with tools for analyzing and visualizing the results.

## Key Features

* **Federated Learning Emulation:** Simulate federated learning with multiple clients and a central server.
* **Federated Averaging:** Implementation of the standard Federated Averaging algorithm.
* **Batch-wise Gradient Transmission:** Exploration of federated learning where clients transmit gradients per mini-batch.
* **Centralized Learning Baseline:** Code for training a model in a centralized manner for comparison.
* **Mini-Batch Size Comparison:** Scripts to evaluate the impact of different mini-batch sizes in federated settings.
* **Training Method Comparison:** Tools to compare the performance of centralized and federated learning approaches.
* **Result Analysis and Visualization:** A Jupyter Notebook (`plotting.ipynb`) for analyzing and plotting training metrics.

## Getting Started

### Prerequisites

* Python 3.x
* TensorFlow (`pip install tensorflow`)
* NumPy (`pip install numpy`)
* Pandas (`pip install pandas`)
* Jupyter (`pip install jupyter`)

You are recommended to use a virtual environment (e.g., `venv` or `conda`) to manage dependencies.

### Installation

1.  **(For future reference when sharing):** Clone the repository:
    ```bash
    git clone <repository_url>
    cd FederatedLearning
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running Examples

Navigate to the repository root and run the following commands to execute the different training scripts:

* **Centralized Learning:**
    ```bash
    python train/centralized_learning.py
    ```
* **Federated Averaging:**
    ```bash
    python train/simple_federated_emulation_per_average.py
    ```
* **Batch-wise Federated Learning:**
    ```bash
    python train/simple_federated_emulation_per_batch.py
    ```
* **Comparing Mini-Batch Sizes:**
    ```bash
    python experiments/comparing_mini_batch_size.py
    ```
* **Comparing Training Methods:**
    ```bash
    python experiments/comparing_training_methods.py
    ```

## Directory Structure
```
FederatedLearning/
├── experiments/
│   ├── comparing_mini_batch_size.py    # Script to compare mini-batch sizes
│   └── comparing_training_methods.py   # Script to compare training methods
├── train/
│   ├── centralized_learning.py        # Centralized learning implementation
│   ├── simple_federated_emulation_per_average.py # Federated Averaging implementation
│   └── simple_federated_emulation_per_batch.py   # Batch-wise Federated Learning
├── fed-learning/                         # Your virtual environment (should be ignored by Git)
├── images/                               # (Optional) Directory for relevant images
├── outputs/                              # Directory for storing training outputs (e.g., models)
├── results/                              # Directory for saving experiment results (e.g., CSVs)
├── README.md                             # This file
├── LICENSE                               # The project's license
├── .gitignore                            # Specifies intentionally untracked files that Git should ignore
├── plotting.ipynb                        # Jupyter Notebook for result analysis and plotting
└── requirements.txt                      # List of Python dependencies
```

## Usage

The Python scripts in the `experiments/train/` directory can be run directly to perform the respective training experiments. You can modify the parameters within these scripts (e.g., number of clients, number of rounds, batch size) to conduct different experiments.

The `plotting.ipynb` notebook can be used to load the result files (typically CSVs saved in the `results/` directory) and generate visualizations of the training metrics (e.g., accuracy, loss over rounds).

## Results

Experiment results, such as training metrics, are typically saved as CSV files in the `results/` directory. The filenames often include information about the experiment parameters.

## Plotting

The `plotting.ipynb` Jupyter Notebook provides tools for analyzing the data in the `results/` directory. You can use it to create plots comparing the performance of different federated learning strategies, batch sizes, or against centralized learning.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Authors/Maintainers

* Gabryel Medeiros de Oliveira
* Lisandro Lovisolo
* Guilherme Lucio Abelha Mota

---
*Last updated: April 10, 2025*