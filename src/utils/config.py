import argparse


def parse_args():
    """
    Parse command line arguments for configuration.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Naive Bayes Classifier for Income Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/adult.csv",
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "--use-uci-split",
        action="store_true",
        help="Use UCI official train/test split (requires adult.data and adult.test)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio (only used if not using UCI split)"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing parameter"
    )
    parser.add_argument(
        "--var-epsilon",
        type=float,
        default=1e-6,
        help="Minimum variance floor for Gaussian features"
    )
    
    parser.add_argument(
        "--force-categorical",
        type=str,
        default="",
        help="Comma-separated list of column names to force as categorical"
    )
    parser.add_argument(
        "--force-numeric",
        type=str,
        default="",
        help="Comma-separated list of column names to force as numeric"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=0,
        help="Number of folds for cross-validation (0 to disable)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["hybrid", "gaussian", "multinomial", "bernoulli"],
        default="hybrid",
        help="Naive Bayes model type"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation"
    )
    
    return parser.parse_args()


def get_force_lists(args):
    """
    Parse the forced categorical/numeric column lists from arguments.

    Args:
        args (argparse.Namespace): The parsed arguments.

    Returns:
        tuple: (force_cat, force_num) where each is a list of column names.
    """
    force_cat = []
    force_num = []
    
    if args.force_categorical:
        force_cat = [s.strip() for s in args.force_categorical.split(",") if s.strip()]
    
    if args.force_numeric:
        force_num = [s.strip() for s in args.force_numeric.split(",") if s.strip()]
    
    return force_cat, force_num
