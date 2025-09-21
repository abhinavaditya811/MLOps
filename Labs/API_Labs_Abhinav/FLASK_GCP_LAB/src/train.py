import joblib
import os
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris

def load_enhanced_data():
    """
    Load and enhance the Iris dataset with realistic noise.
    Returns:
        X (numpy.ndarray): Enhanced features.
        y (numpy.ndarray): Target values.
        feature_names (list): Feature names.
    """
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Add realistic noise to make it more challenging
    np.random.seed(42)
    noise_level = 0.15  # 15% noise
    feature_stds = np.std(X, axis=0)
    noise = np.random.normal(0, feature_stds * noise_level, X.shape)
    X_noisy = X + noise
    
    # Ensure measurements remain positive
    X_noisy = np.maximum(X_noisy, 0.1)
    
    print(f"Enhanced Iris dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Added {noise_level*100}% measurement noise for realism")
    
    return X_noisy, y, iris.feature_names

def train_multiple_models(X_train, y_train):
    """
    Train multiple models and return them.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    print("Training multiple models...")
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_and_select_best(models, X_test, y_test, X_full, y_full):
    """
    Evaluate all models and return the best one.
    """
    print("\nModel Evaluation Results:")
    print("=" * 60)
    
    best_accuracy = 0
    best_model = None
    best_name = None
    
    for name, model in models.items():
        # Test set accuracy
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score (more robust)
        cv_scores = cross_val_score(model, X_full, y_full, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\n{name}:")
        print(f"  Test Set Accuracy: {test_accuracy:.3f}")
        print(f"  Cross-Val Score:   {cv_mean:.3f} (+/- {cv_std * 2:.3f})")
        
        # Use cross-validation score to select best model
        if cv_mean > best_accuracy:
            best_accuracy = cv_mean
            best_model = model
            best_name = name
    
    print(f"\nBest Model: {best_name}")
    print(f"Best CV Score: {best_accuracy:.3f}")
    
    return best_model, best_name

def save_models_and_info(best_model, best_name, all_models, scaler):
    """
    Save all models and metadata.
    """
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save best model (for backward compatibility)
    joblib.dump(best_model, "model/model.pkl")
    print(f"Best model ({best_name}) saved as model.pkl")
    
    # Save all models for switching
    joblib.dump(all_models, "model/all_models.pkl")
    print("All models saved as all_models.pkl")
    
    # Save scaler
    joblib.dump(scaler, "model/scaler.pkl")
    print("Scaler saved as scaler.pkl")
    
    # Save model info
    model_info = {
        'best_model_name': best_name,
        'best_model_type': type(best_model).__name__,
        'available_models': list(all_models.keys()),
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'classes': ['setosa', 'versicolor', 'virginica']
    }
    joblib.dump(model_info, "model/model_info.pkl")
    print("Model info saved")

def detailed_evaluation(model, model_name, X_test, y_test):
    """
    Show detailed evaluation of the best model.
    """
    y_pred = model.predict(X_test)
    
    print(f"\nDetailed Evaluation - {model_name}:")
    print("=" * 50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['setosa', 'versicolor', 'virginica']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nFeature Importances:")
        features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        importance = model.feature_importances_
        for i, feature in enumerate(features):
            print(f"{feature}: {importance[i]:.3f}")

def run_training():
    """
    Enhanced training function with multiple models and evaluation.
    """
    print("Loading and enhancing Iris data...")
    X, y, feature_names = load_enhanced_data()
    
    print("Splitting and preprocessing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.fit_transform(X)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    # Train multiple models
    models = train_multiple_models(X_train_scaled, y_train)
    
    # Evaluate and select best model
    best_model, best_name = evaluate_and_select_best(models, X_test_scaled, y_test, X_full_scaled, y)
    
    # Save all models and metadata
    save_models_and_info(best_model, best_name, models, scaler)
    
    # Detailed evaluation
    detailed_evaluation(best_model, best_name, X_test_scaled, y_test)
    
    print("\nTraining completed successfully!")
    print(f"Best model: {best_name}")
    print("Enhanced Iris classification with multiple algorithms")

if __name__ == "__main__":
    run_training()