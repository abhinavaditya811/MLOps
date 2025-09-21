from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import numpy as np
import os
from data import load_data, split_data

def train_multiple_models(X_train, y_train):
    """
    Train multiple models and return the best one.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
    Returns:
        models (dict): Dictionary of trained models.
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

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and return the best one.
    Args:
        models (dict): Dictionary of trained models.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
    Returns:
        best_model: The model with highest accuracy.
        best_name: Name of the best model.
    """
    from sklearn.model_selection import cross_val_score
    
    print("\nModel Evaluation Results:")
    print("=" * 60)
    
    best_accuracy = 0
    best_model = None
    best_name = None
    
    # Get full dataset for cross-validation
    from data import load_data
    X_full, y_full, _ = load_data()
    
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
        print(f"  CV Individual:     {cv_scores}")
        
        # Use cross-validation score to select best model (more reliable)
        if cv_mean > best_accuracy:
            best_accuracy = cv_mean
            best_model = model
            best_name = name
    
    print(f"\nBest Model: {best_name}")
    print(f"Best CV Score: {best_accuracy:.3f}")
    print("\nNote: Cross-validation provides more reliable model comparison")
    
    return best_model, best_name

def save_model_and_scaler(model, model_name, scaler, all_models):
    """
    Save the best model, scaler, and all trained models.
    Args:
        model: Best performing model.
        model_name: Name of the best model.
        scaler: Fitted StandardScaler object.
        all_models: Dictionary of all trained models.
    """
    model_dir = "../model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save best model (for backward compatibility)
    joblib.dump(model, "../model/iris_model.pkl")
    print(f"Best model ({model_name}) saved as iris_model.pkl")
    
    # Save all models for model switching
    joblib.dump(all_models, "../model/all_models.pkl")
    print("All models saved as all_models.pkl")
    
    # Save scaler
    joblib.dump(scaler, "../model/iris_scaler.pkl")
    print("Scaler saved as iris_scaler.pkl")
    
    # Save model info including all model performances
    model_info = {
        'best_model_name': model_name,
        'best_model_type': type(model).__name__,
        'available_models': list(all_models.keys()),
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'classes': ['setosa', 'versicolor', 'virginica']
    }
    joblib.dump(model_info, "../model/model_info.pkl")
    print("Model info saved with all available models")

def detailed_evaluation(model, model_name, X_test, y_test):
    """
    Show detailed evaluation of the best model.
    Args:
        model: Best performing model.
        model_name: Name of the best model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
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

if __name__ == "__main__":
    print("Loading Iris data...")
    X, y, feature_names = load_data()
    
    print("Splitting and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)
    
    print("Training multiple models...")
    models = train_multiple_models(X_train, y_train)
    
    print("Evaluating models...")
    best_model, best_name = evaluate_models(models, X_test, y_test)
    
    print("Saving best model...")
    save_model_and_scaler(best_model, best_name, scaler, models)
    
    print("Detailed evaluation...")
    detailed_evaluation(best_model, best_name, X_test, y_test)
    
    print("\nTraining completed successfully!")
    print(f"Best model: {best_name}")
    print("Enhanced Iris classification with multiple algorithms")