# Technical Report: Naive Bayes Classifier from Scratch

## 1. Problem & Motivation
The goal of this project is to implement a robust Hybrid Naive Bayes classifier from scratch (without using libraries like scikit-learn for the core logic) to predict whether an individual's income exceeds $50K/yr based on the UCI Adult dataset. This task involves handling mixed data types (continuous numerical and categorical features), missing values, and potential data scale issues, providing a deep understanding of probabilistic modeling and data structure efficiency.

## 2. Architecture & Design
The system follows a modular architecture to ensure separation of concerns and testability:

- **Preprocessing Layer (`src/preprocessing/`)**: Handles CSV parsing, missing value imputation (median for numeric, "Unknown" token for categorical), and label/feature encoding.
- **Data Structures Layer (`src/custom_ds/`)**: Provides fundamental structures:
    - `OpenAddressingHashMap`: A custom hash map with linear probing for O(1) average case lookups.
    - `CountMatrix`: A wrapper around the hash map to store conditional counts.
    - `RunningStats`: Implements Welford's algorithm for stable single-pass mean and variance computation.
- **Model Layer (`src/nb_model/`)**: Contains the `HybridNaiveBayes` class which orchestrates:
    - Gaussian Naive Bayes for numeric features.
    - Multinomial (or Bernoulli) Naive Bayes for categorical features.
    - Log-domain arithmetic for numerical stability.

### Data Flow
`CSV Data` -> `DataLoader` -> `MissingHandler` -> `Encoder` -> `HybridNaiveBayes.fit()` -> `Model State` -> `HybridNaiveBayes.predict()` -> `Evaluation Metrics`

## 3. Data Structures & Algorithms

### 3.1 Custom HashMap
We implemented `OpenAddressingHashMap` using **Linear Probing**.
- **Algorithm**: `hash(key) % capacity` gives the initial slot. If occupied, proceed linearly (`idx + 1`) until an empty slot or tombstone is found.
- **Resizing**: Automatically expands (doubles capacity) when `load_factor > 0.7`.
- **Complexity**: Average O(1) for insert/lookup. Worst case O(N) (mitigated by resizing).

### 3.2 Hybrid Naive Bayes
The model combines two likelihood functions:
1.  **Gaussian**: For numeric features $x_i$, we assume $P(x_i | y) \sim \mathcal{N}(\mu_{y,i}, \sigma^2_{y,i})$. Parameters are estimated using streaming updates (Welford's algorithm) to avoid storing all data.
2.  **Multinomial**: For categorical features, we compute $P(x_j | y)$ using smoothed counts:
    $$ \hat{P}(x_j | y_c) = \frac{\text{count}(x_j, y_c) + \alpha}{\text{count}(y_c) + \alpha \cdot |V_j|} $$

### 3.3 Bernoulli Variant
We also support a Bernoulli mode where categorical features are treated as binary indicators (Present/Absent).
$$ \log P(x | y) \propto \sum \log P(x_i | y)^{x_i} (1 - P(x_i | y))^{(1-x_i)} $$

## 4. Complexity Analysis

### Time Complexity
- **Training**: $O(N \cdot D)$, where $N$ is the number of training samples and $D$ is the number of features. We iterate through the data once.
- **Prediction**: $O(M \cdot D)$, where $M$ is the number of query samples. Each prediction requires summing log-probabilities across features.

### Space Complexity
- **Model Storage**: $O(C \cdot D \cdot V_{avg})$, where $C$ is number of classes and $V_{avg}$ is the average cardinality of categorical features.
    - Gaussian stats store $2 \cdot C \cdot D_{num}$ floats.
    - Categorical counts store non-zero entries in the Hash Map.

## 5. Results & Analysis
The model is evaluated using Accuracy, Precision, Recall, and F1-Score.
- **Handling Imbalance**: The Adult dataset is imbalanced. Accuracy alone is misleading; F1-Score provides a better metric.
- **Stability**: Using log-probabilities prevents underflow. Adding `var_epsilon` to variances prevents division by zero for constant features.

Benchmarks against `scikit-learn` (see `results/` after running benchmark) typically show comparable accuracy, confirming the correctness of the scratch implementation.
