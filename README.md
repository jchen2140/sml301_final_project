# Yelp Review Sentiment Analysis

A multi-task deep learning project for sentiment analysis of Yelp business reviews, combining classification and regression in a single neural network architecture.

## Project Overview

This project implements a **Multi-Kernel Convolutional Neural Network (CNN)** that performs two simultaneous tasks:

1. **Classification**: Predicts sentiment category (Poor / Okay / Good / Amazing)
2. **Regression**: Predicts continuous star rating (1.0 - 5.0)

The model is compared against five classic machine learning baselines (Logistic Regression, Naive Bayes, SVM, Random Forest, and Ridge Regression).

## Dataset

- **Source**: Yelp business reviews (`yelp_cleaned_reviews_1000.csv`)
- **Size**: 1,000 preprocessed reviews
- **Split**: 70% training (700), 30% testing (300)
- **Features**: Cleaned review text with associated star ratings

## Files

### Core Model

| File | Description |
|------|-------------|
| `ml_model.py` | Main neural network model with training and evaluation |

#### `ml_model.py` Details

The main model file implements:

- **Multi-Kernel CNN Architecture**:
  - 4 parallel CNN branches with kernel sizes [2, 3, 4, 5]
  - 256 filters per kernel (1,024 total features)
  - Batch normalization for stable training
  - Dropout regularization (40% after conv, 30% after FC)
  - Dual output heads (classification + regression)

- **GloVe Embeddings**: Pre-trained 100-dimensional word vectors from `glove.6B.100d.txt`

- **Multi-Task Learning**: Combined loss function (60% classification + 40% regression)

- **Training**: 15 epochs with Adam optimizer and cosine annealing learning rate scheduler

**Usage**:
```bash
python ml_model.py
```

---

#### `visualize_classification_softmax.py`

Generates visualizations for the classification task, showing how the model's softmax layer produces probability distributions across sentiment classes.

**Outputs**:
- `softmax_distributions_real.png` - Box plots showing softmax probability distributions for each predicted class
- `softmax_confidence_analysis_real.png` - Analysis of prediction confidence (max probability vs. gap to second choice)

**Usage**:
```bash
python visualize_classification_softmax.py
```

---

#### `visualize_regression_mse.py`

Generates visualizations for the regression task, analyzing prediction errors across different star rating ranges.

**Outputs**:
- `regression_mse_analysis_real.png` - MSE breakdown by true rating class with prediction distributions
- `regression_mse_detailed_real.png` - Detailed error analysis including absolute errors and prediction scatter
- `regression_mse_pipeline_real.png` - Regression pipeline diagram showing loss calculation flow
- `regression_samples_real.png` - Sample predictions comparing actual vs. predicted ratings

**Usage**:
```bash
python visualize_regression_mse.py
```

---

#### `visualize_ml_vs_nn_comparison.py`

Trains all models (5 classic ML + 1 neural network) and generates comparative visualizations.

**Models Compared**:
- Logistic Regression (multinomial, L2 regularization)
- Multinomial Naive Bayes (Laplace smoothing)
- Support Vector Machine (linear kernel)
- Random Forest (100 trees, max depth 20)
- Ridge Regression (L2 regularization)
- Neural Network (Multi-Kernel CNN)

**Outputs**:
- `ml_vs_nn_comparison.png` - Side-by-side accuracy and F1-score comparison
- `ml_vs_nn_detailed.png` - Comprehensive metrics including MAE and RMSE
- `model_comparison_metrics.csv` - Full metrics table for all models

**Usage**:
```bash
python visualize_ml_vs_nn_comparison.py
```

---

#### `generate_metrics_summary.py`

**Outputs**:
- `model_comparison_metrics.csv` - CSV file with all model metrics

**Usage**:
```bash
python generate_metrics_summary.py
```

---

## Requirements

```
torch>=1.9.0
tensorflow>=2.4.0
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## How to run

1. **Train the main model**:
   ```bash
   python ml_model.py
   ```

2. **Generate all visualizations**:
   ```bash
   python visualize_classification_softmax.py
   python visualize_regression_mse.py
   python visualize_ml_vs_nn_comparison.py
   python generate_metrics_summary.py
   ```

* Part of README was helped constructed by AI

glove.6B.100d.txt is too large, so see this google drive for the file: https://drive.google.com/file/d/11JbB-XYND_A25EN2TXxOBve1I4OKYIVs/view?usp=sharing
