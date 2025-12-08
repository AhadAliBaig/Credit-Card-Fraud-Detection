# Credit Card Fraud Detection

A machine learning project comparing traditional and modern approaches to detect credit card fraud using Logistic Regression and Neural Networks (MLP), with and without SMOTE balancing techniques.

## Project Overview

This project evaluates four different machine learning models for credit card fraud detection:
- **Logistic Regression (Baseline)** - Traditional ML on imbalanced data
- **Logistic Regression + SMOTE** - Traditional ML with balanced data
- **Multi-Layer Perceptron (Baseline)** - Neural Network on imbalanced data
- **Multi-Layer Perceptron + SMOTE** - Neural Network with balanced data

The dataset contains **284,807 transactions** with only **492 fraudulent cases** (0.17%), making it a highly imbalanced classification problem.

## How to Run the System

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (optional, for cloning the repository)

### Installation Steps

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd "Credit Card Fraud Detection"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn imbalanced-learn jupyter
   ```

4. **Download the dataset**
   - Place `creditcard.csv` in the `data/` folder
   - The dataset should contain 31 columns (Time, V1-V28, Amount, Class)

5. **Open and run the notebook**
   ```bash
   jupyter notebook fraud_detection.ipynb
   ```
   
   Or with JupyterLab:
   ```bash
   jupyter lab fraud_detection.ipynb
   ```

6. **Execute all cells**
   - Run cells sequentially from top to bottom
   - The notebook will automatically:
     - Load and explore the data
     - Preprocess features
     - Train all 4 models
     - Generate visualizations and results

## Key Results

### Performance Comparison

| Model | F1-Score | True Positives | False Positives | False Negatives |
|-------|----------|----------------|-----------------|-----------------|
| **LR Baseline** | 0.675 | 55 | 10 | 43 |
| **LR + SMOTE** | 0.111 | 90 | 1,440 | 8 |
| **MLP Baseline** | **0.790** | 81 | 26 | 17 |
| **MLP + SMOTE** | 0.779 | 81 | 29 | 17 |

### Dataset Statistics

- **Total Transactions**: 284,807
- **Legitimate (Class 0)**: 284,315 (99.83%)
- **Fraudulent (Class 1)**: 492 (0.17%)
- **Features**: 30 (V1-V28 are PCA-transformed, plus Time and Amount)
- **Train/Test Split**: 80/20

## Key Findings

### 1. **Best Performing Model: MLP Baseline (F1-Score: 0.790)**
   - The Multi-Layer Perceptron trained on imbalanced data achieved the highest F1-score
   - Successfully detected **81 out of 98 fraud cases** (82.7% recall)
   - Only **26 false positives**, minimizing customer inconvenience
   - Neural networks handle class imbalance better than traditional methods

### 2. **SMOTE Impact Varies by Model**
   - **For Logistic Regression**: SMOTE significantly **degraded** performance
     - F1-score dropped from 0.675 to 0.111
     - Generated 1,440 false positives (vs. 10 without SMOTE)
     - While it caught more frauds (90 vs. 55), the high false positive rate makes it impractical
   
   - **For MLP**: SMOTE had minimal impact
     - F1-score slightly decreased from 0.790 to 0.779
     - Similar fraud detection (81 cases) but slightly more false positives (29 vs. 26)

### 3. **Neural Networks Outperform Traditional ML**
   - MLP Baseline (0.790) significantly outperformed LR Baseline (0.675)
   - MLP better captures complex patterns in fraudulent transactions
   - More robust to class imbalance without requiring data balancing

### 4. **Practical Implications**
   - **False positives matter**: In fraud detection, false alarms can:
     - Block legitimate customer transactions
     - Create poor user experience
     - Increase customer service costs
   - **Best approach**: MLP Baseline provides optimal balance between:
     - High fraud detection rate (82.7%)
     - Low false positive rate (0.05%)
     - Best overall F1-score

### 5. **Key Takeaway**
   > **Balancing techniques like SMOTE don't always improve performance.** For this highly imbalanced fraud detection problem, training neural networks on the original imbalanced data yielded superior results compared to using SMOTE-balanced data.

### Models Architecture

**Logistic Regression:**
- Default sklearn parameters
- Random state: 42
- Max iterations: 1000

**Multi-Layer Perceptron (Baseline):**
- Input layer: 30 neurons
- Hidden layer 1: 200 neurons (ReLU)
- Hidden layer 2: 100 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid)
- Optimizer: Adam
- Loss: Binary crossentropy
- Epochs: 10
- Batch size: 32

**Multi-Layer Perceptron (SMOTE):**
- Input layer: 30 neurons
- Hidden layer 1: 100 neurons (ReLU)
- Hidden layer 2: 50 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid)
- Same training parameters as baseline

### Preprocessing
- **Feature Scaling**: StandardScaler (mean=0, std=1)
- **Train/Test Split**: 80/20 with random_state=42
- **SMOTE**: Applied only to training data (never to test set)



## 👤 Author

Credit Card Fraud Detection Project

**Note**: Make sure to download the `creditcard.csv` dataset and place it in the `data/` folder before running the notebook.
