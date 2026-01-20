import os
import json
from src.nb_model.naive_bayes import HybridNaiveBayes
from src.preprocessing.data_loader import load_csv, load_uci_split, is_numeric_column, get_column_values, random_split, k_fold_split
from src.preprocessing.missing_handler import MissingHandler
from src.preprocessing.feature_encoder import FeatureEncoder, LabelEncoder
from src.evaluation.metrics import classification_report, get_metrics_dict
from src.evaluation.plots import plot_confusion_matrix, plot_metrics, plot_learning_curve, plot_class_distribution, check_matplotlib
from src.utils.config import parse_args, get_force_lists
from src.utils.logger import Logger
from src.utils.timers import Timer, TimingResults


def run_pipeline(args, logger, timers, header, train_rows, test_rows):
    """
    Execute the full training and evaluation pipeline.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
        logger (Logger): Logger instance.
        timers (TimingResults): Timing tracker.
        header (list): List of column names.
        train_rows (list): Training data rows.
        test_rows (list): Test data rows.

    Returns:
        tuple: (metrics_dict, hashmap_stats)
    """
    if "income" in header:
        label_idx = header.index("income")
    else:
        label_idx = len(header) - 1
    
    feature_idxs = [i for i in range(len(header)) if i != label_idx]
    force_cat_names, force_num_names = get_force_lists(args)
    numeric_idxs, cat_idxs = [], []
    
    for idx in feature_idxs:
        name = header[idx]
        if name in force_cat_names: is_num = False
        elif name in force_num_names: is_num = True
        else:
            col_vals = get_column_values(train_rows, idx)
            is_num = is_numeric_column(col_vals)
        if is_num: numeric_idxs.append(idx)
        else: cat_idxs.append(idx)
            
    with Timer() as t:
        missing_handler = MissingHandler()
        missing_handler.fit(train_rows, numeric_idxs)
        X_num_train = missing_handler.transform_numeric(train_rows, numeric_idxs)
        X_num_test = missing_handler.transform_numeric(test_rows, numeric_idxs)
        train_rows_cat = missing_handler.transform_categorical(train_rows, cat_idxs)
        test_rows_cat = missing_handler.transform_categorical(test_rows, cat_idxs)
        
        feature_encoder = FeatureEncoder(len(cat_idxs))
        X_cat_train = feature_encoder.fit_transform(train_rows_cat)
        X_cat_test = feature_encoder.transform(test_rows_cat)
        cat_cardinalities = feature_encoder.get_cardinalities()
        
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(get_column_values(train_rows, label_idx))
        y_test = label_encoder.transform(get_column_values(test_rows, label_idx), unknown_id=-1)
    timers.add("Preprocessing", t.elapsed)
    
    with Timer() as t:
        model = HybridNaiveBayes(alpha=args.alpha, var_epsilon=args.var_epsilon, model_type=args.model_type)
        model.fit(X_num_train, X_cat_train, y_train, cat_cardinalities, num_classes=label_encoder.num_classes)
    timers.add("Training", t.elapsed)
    
    with Timer() as t:
        preds = model.predict(X_num_test, X_cat_test)
    timers.add("Prediction", t.elapsed)
    
    valid_indices = [i for i, y in enumerate(y_test) if y != -1]
    y_test_valid = [y_test[i] for i in valid_indices]
    preds_valid = [preds[i] for i in valid_indices]
    
    pos_class_idx = 0
    for i, name in enumerate(label_encoder.id_to_label):
        if ">50K" in name:
            pos_class_idx = i
            break
            
    metrics = get_metrics_dict(y_test_valid, preds_valid, label_encoder.id_to_label, positive_class=pos_class_idx)
    stats = model.hashmap_stats()
    
    return metrics, stats


def main():
    """
    Main entry point for the application.
    Parses arguments, runs the pipeline, and reports results.
    """
    args = parse_args()
    logger = Logger(verbose=True)
    timers = TimingResults()
    
    logger.section(f"Naive Bayes - {args.model_type.upper()} Mode")
    
    if args.use_uci_split:
        header, train_rows, test_rows = load_uci_split(os.path.dirname(args.data_path))
        metrics, stats = run_pipeline(args, logger, timers, header, train_rows, test_rows)
    else:
        header, all_rows = load_csv(args.data_path)
        if args.cv > 1:
            folds = k_fold_split(all_rows, args.cv, args.seed)
            logger.info(f"K-Fold Cross Validation: {args.cv} folds")
            cv_metrics = []
            for i, (train_f, test_f) in enumerate(folds):
                logger.info(f"Processing Fold {i+1}/{args.cv}...")
                m, s = run_pipeline(args, logger, timers, header, train_f, test_f)
                cv_metrics.append(m)
            
            avg_acc = sum(m['accuracy'] for m in cv_metrics) / args.cv
            avg_f1 = sum(m['f1_score'] for m in cv_metrics) / args.cv
            logger.section("Average CV Results")
            logger.info(f"Mean Accuracy: {avg_acc:.4f}")
            logger.info(f"Mean F1 Score: {avg_f1:.4f}")
            return
        else:
            train_rows, test_rows = random_split(all_rows, args.test_ratio, args.seed)
            metrics, stats = run_pipeline(args, logger, timers, header, train_rows, test_rows)
    
    logger.section("Results")
    logger.log(classification_report(None, None, metrics['class_names'], metrics=metrics))
    
    if stats:
        logger.info("Hash Map Statistics:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")
            
            
    logger.log("\n" + timers.summary())
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not args.no_plots:
        if check_matplotlib():
            plot_confusion_matrix(
                metrics['confusion_matrix'], 
                metrics['class_names'], 
                os.path.join(args.output_dir, "confusion_matrix.png")
            )
            plot_metrics(
                metrics, 
                os.path.join(args.output_dir, "metrics.png")
            )
            logger.info("Plots saved to output directory")
        else:
            logger.info("Matplotlib/Seaborn not found. Skipping plots.")
            
        # Optional: Learning Curve Generation
        if check_matplotlib() and not args.no_plots:
            logger.info("Generating Learning Curve (this may take a moment)...")
            subsets = [0.2, 0.4, 0.6, 0.8, 1.0]
            accuracies = []
            sizes = []
            
            # Using a simplified approach for learning curve to avoid heavy complexity
            lc_logger = Logger(verbose=False)
            lc_scores = []
            lc_sizes = []
            
            for ratio in subsets:
                n_sub = int(len(train_rows) * ratio)
                if n_sub < 50: continue
                
                sub_train = train_rows[:n_sub]
                # We use the SAME test set for all
                m, _ = run_pipeline(args, lc_logger, TimingResults(), header, sub_train, test_rows)
                lc_scores.append(m['accuracy'])
                lc_sizes.append(n_sub)
                
            plot_learning_curve(lc_sizes, lc_scores, os.path.join(args.output_dir, "learning_curve.png"))
            
            # Class Distribution Plot
            if "income" in header:
                idx = header.index("income")
            else:
                idx = -1
            train_labels = [row[idx] for row in train_rows]
            plot_class_distribution(train_labels, list(set(train_labels)), os.path.join(args.output_dir, "class_distribution.png"))
            
            logger.info("Advanced plots saved.")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "metrics": metrics, 
            "hashmap_stats": stats, 
            "timings": timers.to_dict()
        }, f, indent=2)


if __name__ == "__main__":
    main()
