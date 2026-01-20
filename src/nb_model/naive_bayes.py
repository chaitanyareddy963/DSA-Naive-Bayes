import math

from src.custom_ds.count_matrix import CountMatrix
from src.custom_ds.running_stats import RunningStats


class HybridNaiveBayes:
    """
    Hybrid Naive Bayes Classifier supporting Gaussian (numeric) and Multinomial/Bernoulli (categorical) features.
    
    Attributes:
        alpha (float): Laplace smoothing parameter.
        var_epsilon (float): Small value added to variance for numerical stability.
        model_type (str): Type of model ('hybrid', 'gaussian', 'multinomial', 'bernoulli').
        class_counts (list): Count of samples per class.
        class_log_priors (list): Log prior probabilities for each class.
        num_means (list): Means of numeric features per class.
        num_vars (list): Variances of numeric features per class.
        counts (CountMatrix): Storage for categorical feature counts.
    """
    def __init__(self, alpha=1.0, var_epsilon=1e-6, model_type="hybrid"):
        """
        Initialize the Hybrid Naive Bayes model.

        Args:
            alpha (float): Smoothing parameter (default 1.0).
            var_epsilon (float): Variance smoothing (default 1e-6).
            model_type (str): One of 'hybrid', 'gaussian', 'multinomial', 'bernoulli'.
                              'hybrid' uses Gaussian for numeric and Multinomial for categorical.
        """
        self.alpha = alpha
        self.var_epsilon = var_epsilon
        self.model_type = model_type

        self.class_counts = []
        self.class_log_priors = []
        self.num_means = []
        self.num_vars = []
        self.cat_cardinalities = []
        self.counts = None

    def fit(self, X_num, X_cat, y, cat_cardinalities, num_classes=None):
        """
        Fit the model to training data.

        Args:
            X_num (list): List of numeric feature vectors.
            X_cat (list): List of categorical feature vectors.
            y (list): List of labels.
            cat_cardinalities (list): Cardinality of each categorical feature.
            num_classes (int): Total number of classes (optional).
        """
        n = len(y)
        if n == 0:
            raise ValueError("Empty training data")

        use_num = self.model_type in ["hybrid", "gaussian"]
        use_cat = self.model_type in ["hybrid", "multinomial", "bernoulli"]
        is_bernoulli = self.model_type == "bernoulli"

        num_features = len(X_num[0]) if (X_num and use_num) else 0
        cat_features = len(X_cat[0]) if (X_cat and use_cat) else 0
        self.cat_cardinalities = list(cat_cardinalities) if use_cat else []

        if num_classes is None:
            class_count = 0
            for c in y:
                if c + 1 > class_count:
                    class_count = c + 1
        else:
            class_count = int(num_classes)

        self.class_counts = [0] * class_count
        num_stats = [[RunningStats() for _ in range(num_features)] for _ in range(class_count)]
        self.counts = CountMatrix(capacity=64, max_load_factor=0.7) if use_cat else None

        for i in range(n):
            c = y[i]
            self.class_counts[c] += 1

            if num_features:
                row_num = X_num[i]
                for j in range(num_features):
                    num_stats[c][j].add(row_num[j])

            if cat_features:
                row_cat = X_cat[i]
                for j in range(cat_features):
                    val = row_cat[j]
                    if is_bernoulli:
                        # Treat as binary: >0 (Present) is 1, 0 (Unknown/Absent) is 0
                        val = 1 if val > 0 else 0
                    self.counts.increment(c, j, val, 1)

        total = sum(self.class_counts)
        self.class_log_priors = []
        for c in range(class_count):
            if self.class_counts[c] == 0:
                self.class_log_priors.append(-1e18)
            else:
                self.class_log_priors.append(math.log(self.class_counts[c] / total))

        self.num_means = [[0.0] * num_features for _ in range(class_count)]
        self.num_vars = [[0.0] * num_features for _ in range(class_count)]
        for c in range(class_count):
            for j in range(num_features):
                m = num_stats[c][j].mean
                v = num_stats[c][j].variance()
                if v < self.var_epsilon:
                    v = self.var_epsilon
                self.num_means[c][j] = m
                self.num_vars[c][j] = v

    def _gaussian_log_pdf(self, x, mean, var):
        """Compute log PDF of Gaussian distribution."""
        return -0.5 * math.log(2.0 * math.pi * var) - ((x - mean) ** 2) / (2.0 * var)

    def predict_one(self, x_num, x_cat):
        """
        Predict class for a single sample.
        
        Args:
            x_num: Numeric features vector.
            x_cat: Categorical features vector.
            
        Returns:
            Predicted class index.
        """
        class_count = len(self.class_counts)
        use_num = self.model_type in ["hybrid", "gaussian"]
        use_cat = self.model_type in ["hybrid", "multinomial"]
        use_bernoulli = self.model_type == "bernoulli"

        log_scores = [0.0] * class_count
        for c in range(class_count):
            score = self.class_log_priors[c]

            if use_num:
                for j in range(len(self.num_means[c])):
                    score += self._gaussian_log_pdf(x_num[j], self.num_means[c][j], self.num_vars[c][j])

            if use_cat:
                for j in range(len(self.cat_cardinalities)):
                    V = self.cat_cardinalities[j]
                    count = self.counts.get_count(c, j, x_cat[j])
                    denom = self.class_counts[c] + self.alpha * V
                    score += math.log((count + self.alpha) / denom)
            
            if use_bernoulli:
                for j in range(len(x_cat)):
                    # Bernoulli: P(x_j | y) = p^x (1-p)^(1-x)
                    # p = (count(1) + alpha) / (count(total) + 2*alpha)
                    val = 1 if x_cat[j] > 0 else 0
                    
                    count_1 = self.counts.get_count(c, j, 1)
                    denom = self.class_counts[c] + self.alpha * 2
                    
                    prob_1 = (count_1 + self.alpha) / denom
                    
                    if val == 1:
                        score += math.log(prob_1)
                    else:
                        score += math.log(1.0 - prob_1)

            log_scores[c] = score

        best_c = 0
        best_score = log_scores[0]
        for c in range(1, class_count):
            if log_scores[c] > best_score:
                best_score = log_scores[c]
                best_c = c
        return best_c

    def predict(self, X_num, X_cat):
        """Predict classes for multiple samples."""
        n = len(X_cat) if X_cat else len(X_num)
        preds = []
        for i in range(n):
            x_num = X_num[i] if X_num else []
            x_cat = X_cat[i] if X_cat else []
            preds.append(self.predict_one(x_num, x_cat))
        return preds

    def hashmap_stats(self):
        """Return statistics of the underlying count matrix hashmap."""
        if self.counts:
            return self.counts.map.stats()
        return None

