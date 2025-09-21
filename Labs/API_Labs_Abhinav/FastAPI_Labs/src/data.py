import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    
    print(f"Iris dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {iris.target_names}")
    
    return X, y, feature_names

def split_data(X, y):
    np.random.seed(42)
    noise_level = 0.15
    
    feature_stds = np.std(X, axis=0)
    noise = np.random.normal(0, feature_stds * noise_level, X.shape)
    X_noisy = X + noise
    
    X_noisy = np.maximum(X_noisy, 0.1)
    
    X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.4, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    print(f"Added {noise_level*100}% measurement noise for realism")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def get_feature_info():
    feature_info = {
        'sepal_length': {'min': 4.3, 'max': 7.9, 'default': 5.8, 'help': 'Sepal length in cm'},
        'sepal_width': {'min': 2.0, 'max': 4.4, 'default': 3.1, 'help': 'Sepal width in cm'},
        'petal_length': {'min': 1.0, 'max': 6.9, 'default': 3.8, 'help': 'Petal length in cm'},
        'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2, 'help': 'Petal width in cm'}
    }
    return feature_info