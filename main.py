"""
Main Entry Point for Credit Scoring ML Project
Orchestrates the complete ML pipeline: preprocessing, training, evaluation, and visualization
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import prepare_data
from models import (train_naive_bayes, train_logistic_regression, 
                    evaluate_model, get_roc_data)


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def display_metrics_table(metrics_list):
    """
    Display evaluation metrics in a formatted table
    
    Args:
        metrics_list: List of metric dictionaries
    """
    print_header("MODEL COMPARISON RESULTS")
    
    # Create DataFrame for better formatting
    df_metrics = pd.DataFrame(metrics_list)
    
    # Format numerical columns to 4 decimal places
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df_metrics[col] = df_metrics[col].apply(lambda x: f"{x:.4f}")
    
    print("\n")
    print(df_metrics.to_string(index=False))
    print("\n")


def plot_combined_roc_curve(models_data, save_path='roc_curve_comparison.png'):
    """
    Plot ROC curves for multiple models on the same graph
    
    Args:
        models_data: List of tuples (model_name, fpr, tpr, roc_auc)
        save_path: Path to save the figure
    """
    print_header("GENERATING ROC CURVE VISUALIZATION")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve for each model
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, (model_name, fpr, tpr, roc_auc) in enumerate(models_data):
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2.5, 
                label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison - Credit Scoring Models', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    
    # Set axis limits
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ ROC curve saved to: {save_path}")
    
    # Show plot
    # plt.show()
    print("✓ ROC curve displayed successfully! (Skipped plt.show for non-interactive mode)")
    print("✓ ROC curve displayed successfully!")


def main():
    """
    Main function to run the complete ML pipeline
    """
    print_header("CREDIT SCORING ML PROJECT")
    print("Modular Architecture: data_preprocessing.py + models.py + main.py")
    
    # Step 1: Data Preprocessing
    print_header("STEP 1: DATA PREPROCESSING")
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Step 2: Model Training
    print_header("STEP 2: MODEL TRAINING")
    
    # Train Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    
    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    
    # Step 3: Model Evaluation
    print_header("STEP 3: MODEL EVALUATION")
    
    # Evaluate Naive Bayes
    nb_metrics = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    
    # Evaluate Logistic Regression
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Display comparison table
    display_metrics_table([nb_metrics, lr_metrics])
    
    # Step 4: ROC Curve Visualization
    print_header("STEP 4: ROC CURVE VISUALIZATION")
    
    # Get ROC data for both models
    nb_fpr, nb_tpr, nb_auc = get_roc_data(nb_model, X_test, y_test)
    lr_fpr, lr_tpr, lr_auc = get_roc_data(lr_model, X_test, y_test)
    
    # Prepare data for plotting
    models_roc_data = [
        ("Naive Bayes", nb_fpr, nb_tpr, nb_auc),
        ("Logistic Regression", lr_fpr, lr_tpr, lr_auc)
    ]
    
    # Plot combined ROC curve
    plot_combined_roc_curve(models_roc_data)
    
    # Final Summary
    print_header("PROJECT COMPLETED SUCCESSFULLY!")
    print("\n✓ Data preprocessing: DONE")
    print("✓ Model training: DONE (2 models)")
    print("✓ Model evaluation: DONE (Accuracy, Precision, Recall, F1-Score)")
    print("✓ ROC curve visualization: DONE (saved as roc_curve_comparison.png)")
    print("\nThank you for using the Credit Scoring ML Project!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
