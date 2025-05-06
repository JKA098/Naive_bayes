# 🧮 Naïve Bayes Classifier for Categorical Data

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Focus](https://img.shields.io/badge/focus-Probabilistic%20Inference-blueviolet)
![Theme](https://img.shields.io/badge/theme-Naive%20Bayes-brightgreen)
![Data](https://img.shields.io/badge/data%20type-Categorical-lightgrey)
![Statistics](https://img.shields.io/badge/statistical%20tests-Posterior%20Probability-blue)
![ML](https://img.shields.io/badge/algorithm-Naive%20Bayes-orange)
![Framework](https://img.shields.io/badge/framework-Custom%20Code-informational)
![Notebook](https://img.shields.io/badge/editor-Jupyter-orange)

---

## 📌 Overview

This project implements a **Naïve Bayes classifier from scratch**, using only built-in libraries for file I/O and performance evaluation. It supports **any dataset with categorical features**.

All classification and probability inference steps are calculated explicitly, without relying on machine learning libraries such as `scikit-learn`. The goal is to apply **Bayes' theorem** and understand the inner mechanics of probabilistic classifiers.

---

## 🎯 Objectives

1. Implement the Naïve Bayes classifier using conditional probabilities.
2. Ensure compatibility with **any number of categorical features**.
3. Accurately predict classes on test data and evaluate performance.


---

## 🚧 Deliverables

* [x] `Naive bayes.ipynb`: Full implementation of the Naïve Bayes algorithm in a Jupyter Notebook.
* [x] Categorical `.csv` dataset with labeled classes.
* [x] Internal logic to:

  * Load and preprocess the data.
  * Split data into training and test sets.
  * Compute prior and conditional probabilities.
  * Classify using **maximum a posteriori** inference.
  * Report accuracy.

---

## 🧠 Algorithm Summary

### 🔹 Assumptions

* Features are **categorical** and **conditionally independent** given the class.
* No missing values are assumed.

### 🔹 Steps Implemented

1. **Data Loading**: Accepts any `.csv` file where the final column is the class label.
2. **Data Splitting**: 80% for training, 20% for testing (using `train_test_split`).
3. **Probability Estimation**:

   * **Prior**: P(class)
   * **Conditional**: P(feature value | class)
   * **Posterior**: Product of conditionals × prior
4. **Prediction**: Class with the highest posterior is assigned.
5. **Evaluation**: Computes test set **accuracy** using built-in metrics.

---

## 🧪 Sample Execution (Notebook Cells)

```python
# Load and split dataset
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("your_dataset.csv")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Train and classify
# ... [probability estimation and inference logic] ...

# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Test Set Accuracy:", accuracy_score(y_true, y_pred))
```

---

## 🤔 Observations

* Accuracy depends heavily on the quality of conditional probabilities, which in turn rely on the amount and diversity of training data.
* Feature independence is a simplifying assumption, but still effective in many real-world scenarios.
* Implementation is general-purpose: no hardcoding of feature names or class values.



