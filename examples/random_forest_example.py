"""
Example usage of the ForestShield RandomForestClassifier.
"""

import numpy as np
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier

# Import our implementation
import sys
import os

sys.path.append(os.path.abspath(".."))
from forest_shield import RandomForestClassifier, SelectFromModel

# Generate a random classification dataset
print("Generating dataset...")
X, y = make_classification(
    n_samples=20000,  # Moderate dataset size for testing
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_repeated=0,
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

# Train and evaluate our RandomForestClassifier
print("\n--- ForestShield RandomForestClassifier ---")
start_time = time.time()
rf = RandomForestClassifier(
    n_estimators=3, max_depth=10, n_jobs=-1, random_state=42, verbose=2
)
rf.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"Training time: {train_time:.4f} seconds")

start_time = time.time()
y_pred = rf.predict(X_test)
predict_time = time.time() - start_time
accuracy = np.mean(y_pred == y_test)
print(f"Prediction time: {predict_time:.4f} seconds")
print(f"Accuracy: {accuracy:.4f}")

# Get feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature ranking:")
for i, idx in enumerate(indices[:5]):
    print(f"{i+1}. Feature {idx}: {importances[idx]:.4f}")

# Feature selection
print("\n--- Feature Selection with SelectFromModel ---")
selector = SelectFromModel(rf, threshold="mean")
selector.fit(X_train, y_train)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of selected features: {X_train_selected.shape[1]}")

# Train a new model on the selected features
rf_selected = RandomForestClassifier(
    n_estimators=3, n_jobs=-1, random_state=42, verbose=2
)
rf_selected.fit(X_train_selected, y_train)

# Evaluate the model
y_pred_selected = rf_selected.predict(X_test_selected)
accuracy_selected = np.mean(y_pred_selected == y_test)
print(f"Accuracy with selected features: {accuracy_selected:.4f}")

# Compare with scikit-learn's RandomForestClassifier
try:
    print("\n--- Scikit-learn RandomForestClassifier (for comparison) ---")
    start_time = time.time()
    sklearn_rf = SklearnRandomForestClassifier(
        n_estimators=3, max_depth=10, n_jobs=-1, random_state=42, verbose=2
    )
    sklearn_rf.fit(X_train, y_train)
    sklearn_train_time = time.time() - start_time
    print(f"Training time: {sklearn_train_time:.4f} seconds")

    start_time = time.time()
    sklearn_y_pred = sklearn_rf.predict(X_test)
    sklearn_predict_time = time.time() - start_time
    sklearn_accuracy = np.mean(sklearn_y_pred == y_test)
    print(f"Prediction time: {sklearn_predict_time:.4f} seconds")
    print(f"Accuracy: {sklearn_accuracy:.4f}")

    # Performance comparison
    print("\n--- Performance Comparison ---")
    print(f"Training speedup: {sklearn_train_time / train_time:.2f}x")
    print(f"Prediction speedup: {sklearn_predict_time / predict_time:.2f}x")
    print(f"Accuracy difference: {(accuracy - sklearn_accuracy) * 100:.4f}%")
except ImportError:
    print("Scikit-learn not installed. Skipping comparison.")
