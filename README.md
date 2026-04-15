# Fake vs Real News Classification using Machine Learning (NLP)

## Overview
This project implements an end-to-end machine learning pipeline for classifying news articles as real or fake using Natural Language Processing (NLP) techniques and multiple supervised learning models.

The workflow includes text preprocessing, TF-IDF feature extraction, dimensionality reduction using PCA, model training, hyperparameter tuning, and comprehensive evaluation using multiple validation strategies.

---

## Why I Built This
I built this project to:
- Apply NLP techniques to a real-world classification problem
- Compare multiple machine learning models on text data
- Understand the impact of preprocessing, feature extraction, and dimensionality reduction
- Evaluate models using both train-test splits and cross-validation methods

---

## Dataset
- Source: Kaggle Fake News Detection Dataset  
- Link: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets  
- Files used:
  - True.csv (real news)
  - Fake.csv (fake news)

Each news article is stored as a single line of text.

---

## Methodology
1. Data Preprocessing:
   - Lowercasing text
   - Removing special characters and punctuation
   - Splitting text into words using basic tokenization
   - Label encoding (real = 1, fake = 0)

2. Feature Extraction:
   - TF-IDF Vectorization (max 1000 features)

3. Dimensionality Reduction:
   - PCA applied to reduce TF-IDF features to 2 components (used for experimentation)

---

## Models Implemented

Naïve Bayes:
- Gaussian Naïve Bayes (on PCA-reduced features)
- Multinomial Naïve Bayes
- Bernoulli Naïve Bayes

Neural Network:
- MLPClassifier with GridSearchCV
- Backpropagation-based learning
- Hyperparameter tuning (hidden layers, activation, learning rate)

Support Vector Machine (SVM):
- Linear Kernel
- RBF Kernel

---

## Evaluation Strategy
Each model is evaluated using:

Train-Test Split:
- 70/30 random split
- 70/30 stratified split

Cross Validation:
- 10-fold K-Fold (random shuffle)
- 10-fold Stratified K-Fold

Metrics:
- Accuracy
- F1 Score

---

## Tech Stack
- Python
- Jupyter Notebook
- pandas
- numpy
- scikit-learn

---

## How to Run
1. Install dependencies:
pip install numpy pandas scikit-learn

2. Download dataset from Kaggle:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

3. Place dataset files in the same directory:
True.csv
Fake.csv

4. Launch Jupyter Notebook:
jupyter notebook

5. Open and run:
fake_news_classification.ipynb

Run all cells in order to execute the full pipeline.

---

## Output
The notebook generates:
- Accuracy scores for all models
- F1 scores for all models
- Cross-validation results
