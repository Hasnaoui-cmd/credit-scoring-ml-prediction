"""
Models Module for Credit Scoring ML Project
Contains model training and evaluation functions
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import numpy as np


def train_naive_bayes(X_train, y_train):
    """
    Train a Gaussian Naive Bayes classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained GaussianNB model
    """
    print("\nTraining Naive Bayes model...")
    model = GaussianNB()
    model.fit(X_train, y_train)
    print("Naive Bayes training completed!")
    return model


def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Train a Logistic Regression classifier
    
    Args:
        X_train: Training features
        y_train: Training labels
        max_iter: Maximum number of iterations
        
    Returns:
        Trained LogisticRegression model
    """
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    print("Logistic Regression training completed!")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for display
        
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"\nEvaluating {model_name}...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    print(f"{model_name} evaluation completed!")
    return metrics


def get_roc_data(model, X_test, y_test):
    """
    Get ROC curve data for a model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        
    Returns:
        fpr, tpr, roc_auc (False Positive Rate, True Positive Rate, AUC score)
    """
    # Get prediction probabilities
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        # For models without predict_proba, use decision_function
        y_pred_proba = model.decision_function(X_test)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


if __name__ == "__main__":
    print("This module contains model training and evaluation functions.")
    print("Import this module in main.py to use the functions.")
