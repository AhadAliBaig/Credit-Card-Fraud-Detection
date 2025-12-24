# Credit Card Fraud Detection

Dataset link: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

A machine learning project comparing traditional and modern approaches to detect credit card fraud using Logistic Regression and Neural Networks (MLP), with and without SMOTE balancing techniques. [web:1]

## Project Overview

This project evaluates four different machine learning models for credit card fraud detection:
- **Logistic Regression (Baseline)** – Traditional ML on imbalanced data
- **Logistic Regression + SMOTE** – Traditional ML with balanced data
- **Multi-Layer Perceptron (Baseline)** – Neural Network on imbalanced data
- **Multi-Layer Perceptron + SMOTE** – Neural Network with balanced data

The dataset contains **284,807 transactions** with only **492 fraudulent cases** (0.17%), making it a highly imbalanced classification problem. [web:1]

## How to Run the System

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git (optional, for cloning the repository)

### Installation Steps

1. **Clone or download this repository**
git clone <repository-url>
cd "Credit Card Fraud Detection"

2. **Create a virtual environment (recommended)**

3. **Install required packages**

4. **Download the dataset**
- Download `creditcard.csv` from Kaggle and place it in the `data/` folder
- The dataset should contain 31 columns (Time, V1–V28, Amount, Class). 

1. **Open and run the notebook**

2. **Execute all cells**
- Run cells sequentially from top to bottom
- The notebook will automatically:
  - Load and explore the data
  - Preprocess features
  - Train all 4 models
  - Generate visualizations and results

## Key Results

### Performance Comparison

| Model                | F1-Score | True Positives | False Positives | False Negatives |
|----------------------|----------|----------------|-----------------|-----------------|
| **LR Baseline**      | 0.675    | 55             | 10              | 43              |
| **LR + SMOTE**       | 0.111    | 90             | 1,440           | 8               |
| **MLP Baseline**     | **0.790**| 81             | 26              | 17              |
| **MLP + SMOTE**      | 0.779    | 81             | 29              | 17              |

### Dataset Statistics

- **Total Transactions**: 284,807  
- **Legitimate (Class 0)**: 284,315 (99.83%)  
- **Fraudulent (Class 1)**: 492 (0.17%)  
- **Features**: 30 (V1–V28 are PCA-transformed, plus Time and Amount)  
- **Train/Test Split**: 80/20. [web:1]

## Key Findings

### 1. Best Performing Model: MLP Baseline (F1-Score: 0.790)

- The Multi-Layer Perceptron trained on imbalanced data achieved the highest F1-score.
- It detected **81 out of 98 fraud cases** (82.7% recall) with only **26 false positives**, minimizing customer inconvenience.
- Neural networks handled the class imbalance better than the traditional Logistic Regression model.

### 2. SMOTE Impact Varies by Model

- **Logistic Regression + SMOTE**:
- F1-score dropped from 0.675 to 0.111.
- False positives increased from 10 to 1,440, making it impractical despite catching more frauds (90 vs. 55).
- **MLP + SMOTE**:
- F1-score slightly decreased from 0.790 to 0.779.
- Fraud detection stayed the same (81 cases), with a small increase in false positives (29 vs. 26).

### 3. Neural Networks Outperform Traditional ML

- The **MLP Baseline (0.790)** significantly outperformed the **LR Baseline (0.675)**.
- The MLP captured more complex patterns in fraudulent transactions and was more robust to the imbalanced data without requiring SMOTE.

### 4. Practical Implications

- **False positives matter**:
- They can block legitimate transactions, harm user experience, and increase customer service workload.
- **Best approach in this project**:
- The MLP Baseline offered the best balance between:
 - High fraud detection rate (82.7% recall)
 - Low false positive rate (~0.05% of transactions)
 - Highest overall F1-score
 

## Preprocessing

- **Feature Scaling**: StandardScaler (mean = 0, std = 1)  
- **Train/Test Split**: 80/20 with `random_state=42`  
- **SMOTE**: Applied only to the training data (never to the test set)  



**Note**: Make sure to download the `creditcard.csv` dataset from Kaggle and place it in the `data/` folder before running the notebook. 
