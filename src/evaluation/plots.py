import os

def check_matplotlib():
    """
    Check if matplotlib and seaborn are valid/available.
    
    Returns:
        bool: True if plotting libraries are installed, False otherwise.
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import seaborn as sns
        return True
    except ImportError:
        return False

def plot_confusion_matrix(cm, class_names, output_path):
    """
    Generate and save a confusion matrix heatmap.
    
    Args:
        cm (list): 2D list or array representing confusion matrix.
        class_names (list): List of class names.
        output_path (str): Path to save the plot.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"[WARNING] Failed to plot confusion matrix: {e}")
        return False

def plot_metrics(metrics, output_path):
    """
    Generate and save a bar chart of evaluation metrics.
    
    Args:
        metrics (dict): Dictionary of metric values.
        output_path (str): Path to save the plot.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=values, palette='viridis')
        plt.ylim(0, 1.0)
        plt.title(f"Evaluation Metrics (Positive Class: {metrics['positive_class']})")
        for i, v in enumerate(values):
            plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"[WARNING] Failed to plot metrics: {e}")
        return False

def plot_learning_curve(sizes, scores, output_path):
    """
    Generate and save a learning curve plot.
    
    Args:
        sizes (list): List of training set sizes.
        scores (list): Corresponding accuracy scores.
        output_path (str): Path to save the plot.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        plt.plot(sizes, scores, marker='o', linestyle='-', color='b')
        plt.title('Learning Curve: Accuracy vs Training Size')
        plt.xlabel('Training Set Size (samples)')
        plt.ylabel('Accuracy')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"[WARNING] Failed to plot learning curve: {e}")
        return False

def plot_class_distribution(y_train, class_names, output_path):
    """
    Generate and save a bar chart of class distribution.
    
    Args:
        y_train (list): List of training labels.
        class_names (list): List of class names.
        output_path (str): Path to save the plot.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Count frequencies manually to avoid pandas dep in generic utils if possible, 
        # but here we can assume lists
        from collections import Counter
        counts = Counter(y_train)
        
        # If y_train contains raw labels (strings), use them directly
        # If y_train contains indices, map them using class_names
        labels = []
        for c in counts.keys():
            if isinstance(c, int) and class_names:
                labels.append(class_names[c])
            else:
                labels.append(str(c))
                
        values = list(counts.values())
        
        plt.figure(figsize=(6, 5))
        sns.barplot(x=labels, y=values, palette='pastel')
        plt.title('Class Distribution (Training Set)')
        plt.xlabel('Class')
        plt.ylabel('Count')
        for i, v in enumerate(values):
            plt.text(i, v + 1, str(v), ha='center')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"[WARNING] Failed to plot class distribution: {e}")
        return False

def plot_benchmark_comparison(custom_res, sklearn_res, output_path):
    """
    Generate and save a comparison plot between custom and sklearn models.
    
    Args:
        custom_res (dict): Results from custom model.
        sklearn_res (dict): Results from sklearn model.
        output_path (str): Path to save the plot.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        
        df = pd.DataFrame([
            {'Model': 'Custom NB', 'Metric': 'Accuracy', 'Value': custom_res['acc']},
            {'Model': 'Sklearn NB', 'Metric': 'Accuracy', 'Value': sklearn_res['acc']},
            {'Model': 'Custom NB', 'Metric': 'Time (s)', 'Value': custom_res['time']},
            {'Model': 'Sklearn NB', 'Metric': 'Time (s)', 'Value': sklearn_res['time']}
        ])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.barplot(data=df[df['Metric'] == 'Accuracy'], x='Model', y='Value', ax=axes[0], palette='Blues')
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylim(0, 1.0)
        
        sns.barplot(data=df[df['Metric'] == 'Time (s)'], x='Model', y='Value', ax=axes[1], palette='Reds')
        axes[1].set_title('Total Time (Train + Predict)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True
    except Exception as e:
        print(f"[WARNING] Failed to plot benchmark comparison: {e}")
        return False
