"""
Example usage of the ForestShield SelectFromModel for feature selection.
"""

import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel as SklearnSelectFromModel
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

# Import our implementation
import sys
import os

sys.path.append(os.path.abspath(".."))
from forest_shield import RandomForestClassifier, SelectFromModel

# Generate a random classification dataset with many features
print("Generating dataset with many features...")
X, y = make_classification(
    n_samples=5000,
    n_features=100,
    n_informative=20,
    n_redundant=30,
    n_repeated=10,
    n_classes=2,
    random_state=42,
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Dataset shape: {X.shape}")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train a random forest classifier
print("\n--- Training RandomForestClassifier ---")
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.4f} seconds")

# Evaluate on full feature set
start_time = time.time()
y_pred_full = rf.predict(X_test)
predict_time_full = time.time() - start_time
accuracy_full = np.mean(y_pred_full == y_test)
print(f"Prediction time (full features): {predict_time_full:.4f} seconds")
print(f"Accuracy (full features): {accuracy_full:.4f}")

# Feature selection with different thresholds
thresholds = ["mean", "median", "0.5*mean", "1.5*mean"]

for threshold in thresholds:
    print(f"\n--- Feature Selection with threshold={threshold} ---")

    # Select features
    start_time = time.time()
    selector = SelectFromModel(rf, threshold=threshold)
    selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    selection_time = time.time() - start_time

    print(f"Selection time: {selection_time:.4f} seconds")
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of selected features: {X_train_selected.shape[1]}")

    # Train a new model on the selected features
    rf_selected = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_selected.fit(X_train_selected, y_train)

    # Evaluate the model
    start_time = time.time()
    y_pred_selected = rf_selected.predict(X_test_selected)
    predict_time_selected = time.time() - start_time
    accuracy_selected = np.mean(y_pred_selected == y_test)

    print(f"Prediction time (selected features): {predict_time_selected:.4f} seconds")
    print(f"Accuracy (selected features): {accuracy_selected:.4f}")
    print(f"Speedup: {predict_time_full / predict_time_selected:.2f}x")
    print(f"Accuracy difference: {(accuracy_selected - accuracy_full) * 100:.4f}%")

# Compare with scikit-learn's SelectFromModel
try:
    print("\n--- Scikit-learn SelectFromModel (for comparison) ---")

    # Train scikit-learn RandomForestClassifier
    sklearn_rf = SklearnRandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    )
    sklearn_rf.fit(X_train, y_train)

    # Select features using scikit-learn
    start_time = time.time()
    sklearn_selector = SklearnSelectFromModel(sklearn_rf, threshold="mean")
    sklearn_selector.fit(X_train, y_train)
    X_train_sklearn_selected = sklearn_selector.transform(X_train)
    X_test_sklearn_selected = sklearn_selector.transform(X_test)
    sklearn_selection_time = time.time() - start_time

    print(f"Selection time: {sklearn_selection_time:.4f} seconds")
    print(f"Number of selected features: {X_train_sklearn_selected.shape[1]}")

    # Train a new model on the selected features
    sklearn_rf_selected = SklearnRandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=42
    )
    sklearn_rf_selected.fit(X_train_sklearn_selected, y_train)

    # Evaluate the model
    sklearn_y_pred_selected = sklearn_rf_selected.predict(X_test_sklearn_selected)
    sklearn_accuracy_selected = np.mean(sklearn_y_pred_selected == y_test)

    print(f"Accuracy (scikit-learn selected features): {sklearn_accuracy_selected:.4f}")

    # Compare our implementation with scikit-learn
    print("\n--- Comparison with scikit-learn ---")
    print(f"Selection time speedup: {sklearn_selection_time / selection_time:.2f}x")
    print(
        f"Number of features selected (our implementation): {X_train_selected.shape[1]}"
    )
    print(
        f"Number of features selected (scikit-learn): {X_train_sklearn_selected.shape[1]}"
    )
    print(
        f"Accuracy difference: {(accuracy_selected - sklearn_accuracy_selected) * 100:.4f}%"
    )

except ImportError:
    print("Scikit-learn not installed. Skipping comparison.")
