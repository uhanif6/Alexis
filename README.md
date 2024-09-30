# Alexis: Anomaly-Based Intrusion Detection in FPGA-Enabled Cyber-Physical Systems

## Overview

**Alexis** is an anomaly detection framework designed to secure FPGA-enabled Cyber-Physical Systems (CPS) against malicious bitstream modifications. By leveraging machine learning, specifically autoencoders, Alexis compares FPGA bitstreams to a pre-trained "Golden Reference" to detect anomalies and potential security threats such as hardware Trojans.

The framework has been evaluated using real-world bitstreams, demonstrating its ability to detect anomalies with high accuracy. This repository contains the code and dataset used to validate Alexis' effectiveness.

## Dataset

The dataset used in this study consists of FPGA bitstream data. The bitstreams are categorized into two main groups:
- **Normal Bitstreams**: Bitstreams with no malicious modifications.
- **Anomalous Bitstreams**: Bitstreams injected with various types of anomalies, including hardware Trojans.

You can download the dataset used for training and testing the model here: [Dataset Link](https://dx.doi.org/10.21227/aqc1-dv65)

## Features

- **Anomaly Detection**: Identifies bitstream anomalies using autoencoders and other machine learning models.
- **FPGA Bitstream Analysis**: Compares the current bitstream to a "Golden Reference" for integrity assurance.
- **Modular Design**: Easily extendable to include additional anomaly detection techniques.
- **Real-World Validation**: Evaluated on real FPGA bitstream data.

## Installation

To run this project, you'll need to clone the repository and install the required dependencies.

### Clone the repository


```bash
git clone https://github.com/uhanif6/Alexis.git
cd Alexis-main
```

## Install dependencies

You can install the required Python libraries by running:

```bash
pip install -r requirements.txt
```
## Performance
### Key Performance Metrics:

- **AUC (Area Under Curve)**: 0.90+
- **MSE Loss**: 0.5439
- **MAE Loss**: 0.5335

## Visualization
The framework generates various plots, including:

Receiver Operating Characteristic (ROC) curve for evaluating the model's anomaly detection performance.
Reconstruction Error Distribution for visualizing how well the model reconstructs normal bitstreams.
You can visualize the performance of the model using the following command:


python visualize.py
Model Architecture
The Alexis model uses a Convolutional Variational Autoencoder (C-VAE) consisting of:

- **Encoder**: Three fully connected layers with ReLU activations.
- **Decoder**: Three fully connected layers for reconstructing the input bitstream.
- **Dropout Layers**: Added to reduce overfitting and improve generalization.
