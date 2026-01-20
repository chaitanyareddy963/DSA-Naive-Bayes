# Naive Bayes Classifier from Scratch

A robust, modular implementation of Naive Bayes for the Adult Income dataset, building all core data structures and algorithms from scratch in Python.

## Project Structure
```
.
├── src/
│   ├── custom_ds/       # OpenAddressingHashMap, RunningStats (Welford's)
│   ├── nb_model/        # HybridNaiveBayes (Gaussian + Multinomial/Bernoulli)
│   ├── preprocessing/   # DataLoader, MissingHandler, FeatureEncoder
│   ├── evaluation/      # Metrics (F1, Confusion Matrix)
│   └── utils/           # Config, Logger, Timers
├── tests/               # Unit and Integration tests
├── benchmark/           # Scikit-learn comparison script
├── results/             # Output metrics and logs
├── data/                # Dataset directory
├── main.py              # Entry point
├── TECHNICAL_REPORT.md  # Detailed algorithmic & complexity analysis
└── README.md
```

## Requirements
- **Python 3.8+**
- **No external dependencies** are required for the core classifier and `main.py`.
- **Optional Dependencies** (only for benchmarking/plotting):
  - `pandas`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

To install optional dependencies for benchmarking:
```bash
pip install -r requirements_benchmark.txt
```

## How to Run

### 1. Basic Run
Run the classification pipeline with default settings (Gaussian for numeric, Multinomial for categorical):
```bash
python3 main.py
```
This will:
- Load `data/adult.csv`
- Train the Hybrid Naive Bayes model
- Evaluate on a 20% test split
- Save results to `results/metrics.json`

### 2. Configuration Options
You can customize the execution using command-line arguments:

```bash
# Force specific model types (e.g. use Bernoulli for categorical)
# Note: Bernoulli mode treats categorical features as binary (Present=1, Absent=0)
python3 main.py --model-type bernoulli

# Force specific feature types
python3 main.py --force-numeric age --force-categorical education

# Use UCI official split (requires adult.data and adult.test in data/)
python3 main.py --use-uci-split

# Change output directory
python3 main.py --output-dir my_results
```

### 3. Run Tests
Execute the full test suite using `unittest` to verify correctness:
```bash
python3 -m unittest discover tests -v
```

### 4. Run Benchmark
Compare performance against scikit-learn (requires `pandas` and `sklearn`). This is for validation purposes only; the core project does not use these libraries.
```bash
python3 benchmark/compare_sklearn.py data/adult.csv
```

## Expected Output
Running `main.py` should output logs to the console indicating progress:
```
[INFO] Loading data from data/adult.csv...
[INFO] Training HybridNaiveBayes model...
[INFO] Evaluating model...
[INFO] Accuracy: 0.842...
```
It will also generate `results/metrics.json` containing detailed performance metrics.

## Troubleshooting

- **FileNotFoundError: data/adult.csv**: Ensure the dataset is present in the `data/` directory.
- **ModuleNotFoundError**: Ensure you are running python from the root of the project.
- **Matplotlib/Seaborn errors**: These are only needed for `benchmark/compare_sklearn.py`. `main.py` runs without them (ensure you don't enable plotting flags if you lack these libs, or install them).

## Technical Details
For a detailed explanation of the algorithms, data structures (HashMap, Welford's Algorithm), and complexity analysis, please refer to the [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
