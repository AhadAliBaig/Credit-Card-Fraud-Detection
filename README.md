#  Credit Card Fraud Detection

This project detects fraudulent credit card transactions using machine learning. The dataset is highly imbalanced, with only **492 fraud cases out of 284,807 transactions**, which makes the problem challenging.

This project compares two models:
- **Logistic Regression (LR)**
- **Neural Network (MLP)**  

Both were tested with and without **SMOTE** to handle class imbalance.

---

##  Setup

- Dataset: Kaggle Credit Card Fraud  
- Train/Test Split: **80/20**  
- Feature Scaling: **StandardScaler**  
- SMOTE applied only on training data  

---

##  Results

| Model        | F1-Score |
|-------------|---------|
| LR           | 0.675 |
| LR + SMOTE   | 0.111 |
| MLP          | **0.790** |
| MLP + SMOTE  | 0.779 |

---

##  Conclusion

The **MLP model performed best**, detecting most fraud cases while keeping false positives low. Logistic Regression struggled, especially after SMOTE, which caused too many normal transactions to be flagged.

Neural networks handled the imbalance better and are more practical for real-world fraud detection systems.

---

##  How to Run

1. Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place it in the `data/` folder.  
3. Install required packages.  
4. Open the notebook and run all cells.
