# Molecular Graph Node-Level Classificationwith PyTorch Geometric

This repository contains a PyTorch Geometric implementation for node-level classificationon molecular graphs. It leverages RDKit for molecular processing and feature extraction, and PyTorch Geometric for building and training graph neural network models.

## Overview

This project aims to predict node-level properties (e.g., atomic charges, contributions) of molecules by training a graph neural network. The pipeline includes:

1.  **Molecular Processing:** Using RDKit to load, sanitize, and featurize molecules.
2.  **Graph Construction:** Converting molecular data into PyTorch Geometric `Data` objects.
3.  **Model Training:** Building and training a graph neural network model for node-level regression.
4.  **Evaluation:** Assessing model performance using standard regression metrics.

## Files Description

-   **`config.py`**: Defines the project configuration, including data paths, model hyperparameters, and RDKit processing steps.
-   **`dataset.py`**: Handles dataset creation and loading, including caching and feature range calculation.
-   **`early_stopping.py`**: Implements early stopping to prevent overfitting during training.
-   **`feature_extractor.py`**: Extracts atomic and bond features from molecules using one-hot encoding.
-   **`main.py`**: The main script for training and evaluating the model.
-   **`models.py`**: Defines the graph neural network model architecture.
-   **`molecule_processor.py`**: Processes molecules into PyTorch Geometric `Data` objects.
-   **`rdkit_processor.py`**: Contains RDKit processing functions and pipeline definitions.
-   **`training.py`**: Implements the training and validation loops, and evaluation metrics.
-   **`data_loader.py`**: Handles the loading and saving of processed data and feature ranges.

## Dependencies

-   Python 3.13.2
-   PyTorch
-   PyTorch Geometric (`torch_geometric`)
-   RDKit (`rdkit`)
-   scikit-learn (`sklearn`)
-   pandas (`pandas`)
-   matplotlib (`matplotlib`)
-   PyYAML (`pyyaml`)
-   numpy (`numpy`)

Install the required packages using pip:

```bash
pip install torch torch-geometric rdkit-pypi scikit-learn pandas matplotlib pyyaml numpy
Setup and Usage
Clone the repository:

Bash
git clone https://github.com/shahram-boshra/n_class.git (or git@github.com:shahram-boshra/n_class.git)
cd n_reg
Prepare your molecular data:

Place your molecular .mol files in the directory specified by config.data.root_dir in config.yaml.
Create an edge target CSV file specified by config.data.edge_target_csv in config.yaml that contains the node-level classificationtargets. The CSV should have an index column (molecule names) and target columns.

Configure config.yaml:
Adjust the data paths, model hyperparameters, and RDKit processing steps as needed.
Ensure the root_dir and edge_target_csv paths are correctly set.
Modify model parameters like hidden_channels, learning_rate, batch_size, and layer types.
Configure RDKit processing steps in rdkit_processing.
Run the training script:

Bash
python main.py
The script will:
Load and process the molecular data.
Split the dataset into training, validation, and test sets.
Train the graph neural network model.
Evaluate the model on the test set.
Save the test targets and predictions as .npy files.
Generate plots of training/validation losses and evaluation metrics.
Log all relevant information to app.log.
Analyze the results:

View the generated plots for training and validation performance.
Analyze the test set metrics printed in the console.
Examine the saved test_targets.npy and test_predictions.npy files for detailed prediction analysis.
Inspect the app.log for training information.
Model Architecture
The graph neural network model (models.py) supports configurable graph convolutional layers, including:

GCNConv
GATConv
SAGEConv
GINConv
GraphConv
TransformerConv
CustomMPLayer (a custom message passing layer)
The architecture consists of multiple convolutional layers, batch normalization, ReLU activation, dropout, and a linear output layer for regression.

Training Details
The model is trained using the Adam optimizer and Huber loss.
Learning rate scheduling (StepLR, ReduceLROnPlateau) and early stopping are implemented to improve training stability and prevent overfitting.
L1 regularization can be enabled via the config file.
The training and validation progress, as well as the test results, will be logged to the app.log file.

Example Configuration (config.yaml)
YAML
data:
  root_dir: 'C:/Chem_Data' # Path to molecular data
  edge_target_csv: 'C:/Chem_Data/edge_target.csv' # Path to edge target CSV
  use_cache: true
  train_split: 0.7
  valid_split: 0.15

model:
  first_layer_type: 'gcn'
  second_layer_type: 'gcn'
  hidden_channels: 128
  learning_rate: 0.001
  batch_size: 32
  dropout_rate: 0.5
  weight_decay: 0.0001
  step_size: 50
  gamma: 0.5
  reduce_lr_factor: 0.5
  reduce_lr_patience: 10
  early_stopping_patience: 20
  early_stopping_delta: 0.0001
  l1_regularization_lambda: 0.0001

rdkit_processing:
  steps: ['HYDROGENATE', 'SANITIZE', 'KEKULIZE', 'EMBED', 'OPTIMIZE']

Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bug fixes, feature requests, or improvements.

License
This project is licensed under the MIT License.