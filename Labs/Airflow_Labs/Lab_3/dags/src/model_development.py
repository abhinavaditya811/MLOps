import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # CHANGED: Import Random Forest instead

# Load data from a CSV file
def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/advertising.csv"))
    return data

# Preprocess the data
def data_preprocessing(data):
    X = data.drop(['Timestamp', 'Clicked on Ad', 'Ad Topic Line', 'Country', 'City'], axis=1)
    y = data['Clicked on Ad']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    num_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']

    # Define a column transformer for preprocessing
    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder='passthrough'
    )

    # Transform the training and testing data
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    return X_train, X_test, y_train.values, y_test.values

# Build and save a Random Forest model  # CHANGED: Updated comment
def build_model(data, filename):
    X_train, X_test, y_train, y_test = data

    # Create and train a Random Forest model  # CHANGED: Updated comment
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)  # CHANGED: Use Random Forest
    rf_clf.fit(X_train, y_train)

    # Ensure the directory exists
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, filename)
    
    # Save the trained model to a file
    pickle.dump(rf_clf, open(output_path, 'wb'))
    
    # ADDED: Print model performance for verification
    train_score = rf_clf.score(X_train, y_train)
    test_score = rf_clf.score(X_test, y_test)
    print(f"Random Forest - Train accuracy: {train_score:.4f}, Test accuracy: {test_score:.4f}")


# Load a saved Random Forest model and evaluate it  # CHANGED: Updated comment
def load_model(data, filename):
    X_train, X_test, y_train, y_test = data
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    # Load the saved model from a file
    loaded_model = pickle.load(open(output_path, 'rb'))

    # Make predictions on the test data and print the model's score
    predictions = loaded_model.predict(X_test)
    print(f"Model score on test data: {loaded_model.score(X_test, y_test)}")

    return predictions[0]


if __name__ == '__main__':
    x = load_data()
    x = data_preprocessing(x)
    build_model(x, 'model.sav')