# ForestShield: HTTP DDoS Detection with Random Forest

ForestShield is a machine learning-based system for detecting HTTP DDoS attacks using Random Forest classification. The project uses the CICIDS 2017 dataset to train a model capable of identifying various types of application layer DDoS attacks.

## Overview

This project aims to train a Random Forest model using the CICIDS 2017 dataset to detect application layer DDoS attacks, with a specific focus on HTTP DDoS. The system includes components for training the model, running a test web server, and performing real-time DDoS detection.

## Components

### 1. Model Training (`train.py`)

The training script processes the CICIDS 2017 dataset to build a Random Forest classifier capable of detecting various types of DDoS attacks. The script:

- Loads and preprocesses the CICIDS 2017 dataset
- Performs feature engineering to extract relevant network traffic patterns
- Trains a Random Forest model optimized for DDoS detection
- Evaluates model performance using precision, recall, and F1 scores
- Saves the trained model to `model/rf.pkl`

The data path is hard-coded in the training script.

### 2. Test Web Server (`server.py`)

A Flask-based web server that acts as a victim for testing HTTP DDoS attacks. The server:

- Provides several endpoints with different response characteristics
- Collects and logs performance metrics during attacks
- Monitors system resources (CPU, memory, network)
- Logs all activity for later analysis

Available endpoints:
- `/` - Basic homepage
- `/echo` - Echo service (POST)
- `/status` - Server status and metrics
- `/heavy` - CPU-intensive endpoint
- `/slow` - Endpoint with delayed response

### 3. DDoS Detection (`ddos_detect.py`)

Real-time DDoS detection script that:

- Captures network traffic using tcpdump
- Extracts flow features from captured packets
- Applies the trained Random Forest model to classify traffic
- Alerts when DDoS attacks are detected
- Provides attack type classification

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ForestShield.git
   cd ForestShield
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```
   pip install -e .
   ```

## Usage

### Training the Model

```
python train.py
```

This will train the Random Forest model using the CICIDS 2017 dataset and save it to `model/rf.pkl`.

### Running the Test Server

```
python server.py
```

The server will start on http://localhost:8080 by default.

### Running DDoS Detection

```
python ddos_detect.py
```

This will start monitoring network traffic and use the trained model to detect potential DDoS attacks.

## CICFlowMeter Installation

**Important:** You must install CICFlowMeter in your system first to analyze the captured PCAP files. 

It is recommended to install a fixed version: [https://github.com/Latria-Kure/CICFlowMeterIDS](https://github.com/Latria-Kure/CICFlowMeterIDS)

This fixed version:
- Resolves several issues in the original implementation
- Maintains compatibility with the CICIDS 2017 dataset
- Provides pre-built binaries for easy installation

The `ddos_detect.py` script relies on CICFlowMeter to convert captured network traffic (PCAP files) into flow features that can be analyzed by the machine learning model.

## Requirements

- Python 3.6+
- numpy>=1.17.0
- scipy>=1.3.0
- cython>=0.29.0
- scikit-learn>=1.0.0
- flask>=3.0.0
- werkzeug>=3.0.0
- psutil>=5.9.0
- tcpdump (for packet capture)

## Dataset

This project uses the CICIDS 2017 dataset, which contains benign traffic and various attack types including DoS, DDoS, brute force, XSS, SQL injection, infiltration, and botnet activities.

## License

See the [LICENSE](LICENSE) file for details.