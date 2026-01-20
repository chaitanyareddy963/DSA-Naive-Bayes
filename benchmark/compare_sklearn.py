"""
Benchmark script to compare custom implementation against scikit-learn.
Requires: scikit-learn, pandas, numpy
"""
import time
import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.data_loader import load_csv, random_split, is_numeric_column, get_column_values
from src.evaluation.plots import plot_benchmark_comparison, check_matplotlib

def benchmark_sklearn(data_path, test_ratio=0.2, seed=42):
    """
    Run scikit-learn's GaussianNB on the dataset for performance comparison.
    Generates a comparison plot if previous custom results verify.

    Args:
        data_path (str): Path to the CSV dataset.
        test_ratio (float): Fraction of data to use for testing.
        seed (int): Random seed.
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score
    except ImportError as e:
        print("\n" + "!" * 50)
        print("[ERROR] Missing required libraries for benchmarking.")
        print(f"Error: {e}")
        print("Please install: pip install scikit-learn pandas numpy")
        print("!" * 50 + "\n")
        return

    print(f"[BENCHMARK] Loading data from {data_path}...")
    
    header, all_rows = load_csv(data_path)
    train_rows, test_rows = random_split(all_rows, test_ratio, seed)
    
    df_train = pd.DataFrame(train_rows, columns=header)
    df_test = pd.DataFrame(test_rows, columns=header)
    
    print(f"[BENCHMARK] Split: {len(df_train)} train, {len(df_test)} test")
    
    label_col = "income" if "income" in header else header[-1]
    
    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]
    X_test = df_test.drop(columns=[label_col])
    y_test = df_test[label_col]
    
    numeric_cols = []
    cat_cols = []
    
    for col in X_train.columns:
        sample_vals = X_train[col].head(100).tolist()
        if is_numeric_column(sample_vals):
            numeric_cols.append(col)
        else:
            cat_cols.append(col)
            
    print("[BENCHMARK] Preprocessing...")
    
    X_train = X_train.replace("?", np.nan)
    X_test = X_test.replace("?", np.nan)
    
    if numeric_cols:
        imp_num = SimpleImputer(strategy='median')
        X_train[numeric_cols] = imp_num.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = imp_num.transform(X_test[numeric_cols])

    if cat_cols:
        X_train[cat_cols] = X_train[cat_cols].fillna("Unknown")
        X_test[cat_cols] = X_test[cat_cols].fillna("Unknown")
        
        try:
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
            X_test[cat_cols] = enc.transform(X_test[cat_cols])
        except TypeError:
            for col in cat_cols:
                le = LabelEncoder()
                combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
                le.fit(combined)
                X_train[col] = le.transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

    le_y = LabelEncoder()
    y_train_enc = le_y.fit_transform(y_train)
    y_test_enc = le_y.transform(y_test)
    
    pos_label = ">50K"
    pos_class_idx = -1
    for idx, label in enumerate(le_y.classes_):
        if pos_label in label:
            pos_class_idx = idx
            break
    
    if pos_class_idx == -1:
        print(f"Warning: Could not find '{pos_label}' in classes {le_y.classes_}. Using last class.")
        pos_class_idx = len(le_y.classes_) - 1

    print("[BENCHMARK] Training GaussianNB...")
    model = GaussianNB()
    
    start_train = time.perf_counter()
    model.fit(X_train, y_train_enc)
    train_time = time.perf_counter() - start_train
    
    print("[BENCHMARK] Predicting...")
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - start_pred
    
    acc = accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, pos_label=pos_class_idx, average='binary')
    
    print("\n" + "=" * 40)
    print("      SKLEARN BENCHMARK RESULTS       ")
    print("      (Baseline: GaussianNB)          ")
    print("=" * 40)
    print(f"Training Time:    {train_time:.6f} s")
    print(f"Prediction Time:  {pred_time:.6f} s")
    print("-" * 40)
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1 Score (>50K):  {f1:.4f}")
    print(f"F1 Score (>50K):  {f1:.4f}")
    print("=" * 40 + "\n")
    
    # Generate Comparison Plot
    if check_matplotlib():
        print("[BENCHMARK] Checking for custom model results...")
        import json
        custom_metrics_path = os.path.join("results", "metrics.json")
        if os.path.exists(custom_metrics_path):
            with open(custom_metrics_path) as f:
                data = json.load(f)
                cust_acc = data['metrics']['accuracy']
                cust_time = data['timings'].get('Training', 0) + data['timings'].get('Prediction', 0)
                
            custom_res = {'acc': cust_acc, 'time': cust_time}
            sklearn_res = {'acc': acc, 'time': train_time + pred_time}
            
            output_plot = "results/benchmark_comparison.png"
            os.makedirs("results", exist_ok=True)
            plot_benchmark_comparison(custom_res, sklearn_res, output_plot)
            print(f"[INFO] Benchmark comparison plot saved to {output_plot}")
        else:
            print("[INFO] Run main.py first to generate custom results for comparison plot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark against Scikit-learn GaussianNB")
    parser.add_argument("data_path", help="Path to csv data", default="data/adult.csv", nargs="?")
    args = parser.parse_args()
    
    benchmark_sklearn(args.data_path)
