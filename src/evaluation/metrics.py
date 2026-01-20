def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        num_classes (int): Total number of classes.

    Returns:
        list: 2D list representing the confusion matrix where cm[i][j] is 
              the count of samples with true label i and predicted label j.
    """
    cm = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(y_true, y_pred):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            cm[true][pred] += 1
    return cm


def accuracy_from_cm(cm):
    """
    Calculate accuracy from confusion matrix.

    Args:
        cm (list): Confusion matrix.

    Returns:
        float: Accuracy score (0.0 to 1.0).
    """
    total = sum(sum(row) for row in cm)
    if total == 0:
        return 0.0
    correct = sum(cm[i][i] for i in range(len(cm)))
    return correct / total


def precision_score(cm, positive_class):
    """
    Calculate precision for a specific class.
    Precision = TP / (TP + FP)

    Args:
        cm (list): Confusion matrix.
        positive_class (int): Index of the positive class.

    Returns:
        float: Precision score.
    """
    tp = cm[positive_class][positive_class]
    fp = sum(cm[i][positive_class] for i in range(len(cm)) if i != positive_class)
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def recall_score(cm, positive_class):
    """
    Calculate recall (sensitivity) for a specific class.
    Recall = TP / (TP + FN)

    Args:
        cm (list): Confusion matrix.
        positive_class (int): Index of the positive class.

    Returns:
        float: Recall score.
    """
    tp = cm[positive_class][positive_class]
    fn = sum(cm[positive_class][j] for j in range(len(cm)) if j != positive_class)
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.
    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        precision (float): Precision score.
        recall (float): Recall score.

    Returns:
        float: F1 score.
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def classification_report(y_true, y_pred, class_names, positive_class=None, metrics=None):
    """
    Generate a text report showing main classification metrics.

    Args:
        y_true (list): True labels (optional if metrics provided).
        y_pred (list): Predicted labels (optional if metrics provided).
        class_names (list): List of class names.
        positive_class (str): Name of positive class (optional).
        metrics (dict): Pre-calculated metrics dictionary (optional).

    Returns:
        str: Formatted report string.
    """
    num_classes = len(class_names)
    if metrics is None:
        metrics = get_metrics_dict(y_true, y_pred, class_names, positive_class)
    
    cm = metrics["confusion_matrix"]
    acc = metrics["accuracy"]
    prec = metrics["precision"]
    rec = metrics["recall"]
    f1 = metrics["f1_score"]
    pos_name = metrics["positive_class"]
    
    lines = []
    lines.append("=" * 50)
    lines.append("CLASSIFICATION REPORT")
    lines.append("=" * 50)
    lines.append("")
    lines.append("Confusion Matrix:")
    
    header = "Actual\\Pred".ljust(12) + "".join(name[:10].ljust(12) for name in class_names)
    lines.append(header)
    lines.append("-" * len(header))
    
    for i, row in enumerate(cm):
        row_str = class_names[i][:10].ljust(12) + "".join(str(val).ljust(12) for val in row)
        lines.append(row_str)
    
    lines.append("")
    lines.append(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    lines.append(f"Precision (class '{pos_name}'): {prec:.4f}")
    lines.append(f"Recall    (class '{pos_name}'): {rec:.4f}")
    lines.append(f"F1 Score  (class '{pos_name}'): {f1:.4f}")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def get_metrics_dict(y_true, y_pred, class_names, positive_class=None):
    """
    Compute all standard metrics and return as a dictionary.

    Args:
        y_true (list): True labels.
        y_pred (list): Predicted labels.
        class_names (list): List of class names.
        positive_class (int): Index of positive class (defaults to last class).

    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1_score, etc.
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    if positive_class is None:
        positive_class = num_classes - 1
    
    acc = accuracy_from_cm(cm)
    prec = precision_score(cm, positive_class)
    rec = recall_score(cm, positive_class)
    f1 = f1_score(prec, rec)
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "positive_class": class_names[positive_class],
        "confusion_matrix": cm,
        "class_names": class_names
    }
