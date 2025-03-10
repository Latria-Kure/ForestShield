import pandas as pd
import numpy as np
import pickle
from forest_shield.forest import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
)

# Group features by their characteristics
FLOW_METADATA = [
    "Flow ID",
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
]

PACKET_COUNT_FEATURES = [
    "Total Fwd Packet",
    "Total Bwd packets",
    "Total Length of Fwd Packet",
    "Total Length of Bwd Packet",
]

PACKET_LENGTH_STATS = [
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Packet Length Min",
    "Packet Length Max",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "Average Packet Size",
]

FLOW_TIMING_FEATURES = [
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
]

FLAG_FEATURES = [
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWR Flag Count",
    "ECE Flag Count",
]

DERIVED_FEATURES = [
    "Down/Up Ratio",
    "Average Packet Size",
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg",
    "Fwd Packet/Bulk Avg",
    "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg",
    "Bwd Packet/Bulk Avg",
    "Bwd Bulk Rate Avg",
]

SUBFLOW_FEATURES = [
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
]

WINDOW_FEATURES = [
    "FWD Init Win Bytes",
    "Bwd Init Win Bytes",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
]

ACTIVITY_FEATURES = [
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]

# Features to remove (meaningless or redundant)
REMOVE_FEATURES = [
    "Flow ID",
    "Src IP",
    "Dst IP",
    "Src Port",
    "Dst Port",  # Identifiers
    "Protocol",
    "Timestamp",  # Metadata
    "Fwd Segment Size Avg",
    "Bwd Segment Size Avg",  # Redundant with packet length
    "Fwd IAT Total",  # Equal to Flow Duration
    "Bwd PSH Flags",
    "Fwd URG Flags",  # Usually zero
]

USED_LABELS = ["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris"]


def analyze_features(data):
    """
    Analyze features to understand their distributions and relationships
    """
    print("\nFeature Analysis:")

    # Check for constant or near-constant features
    variance = data.var()
    low_variance = variance[variance < 0.01].index.tolist()
    print(f"\nLow variance features (variance < 0.01): {low_variance}")

    # Check for highly correlated features
    corr_matrix = data.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.95:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    print("\nHighly correlated feature pairs (|correlation| > 0.95):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"{feat1} - {feat2}: {corr:.3f}")

    return low_variance, high_corr_pairs


def load_data(file_path):
    """
    Load and perform initial data cleaning
    """
    print(f"\nLoading data from {file_path}")
    data = pd.read_csv(file_path)

    # Basic cleaning
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Analyze missing values
    missing_stats = data.isnull().sum()
    print("\nMissing values per feature:")
    print(missing_stats[missing_stats > 0])

    # Drop rows with missing values (or consider imputation strategy)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)

    # Filter relevant attack types
    data = data[data["Label"].isin(USED_LABELS)]

    # Remove identified meaningless/redundant features
    data.drop(REMOVE_FEATURES, axis=1, inplace=True)

    print(f"\nData shape after cleaning: {data.shape}")
    return data


def create_timing_features(X):
    """
    Create timing-based features that might be useful for DoS detection
    """
    # IAT (Inter-Arrival Time) statistics
    if all(feat in X.columns for feat in ["Fwd IAT Mean", "Bwd IAT Mean"]):
        X["IAT_ratio"] = X["Fwd IAT Mean"] / X["Bwd IAT Mean"].replace(0, 0.1)

    return X


def create_packet_features(X):
    """
    Create packet-based features
    """
    # Packet length ratios
    if all(
        feat in X.columns
        for feat in ["Fwd Packet Length Mean", "Bwd Packet Length Mean"]
    ):
        X["Packet_Length_Ratio"] = X["Fwd Packet Length Mean"] / X[
            "Bwd Packet Length Mean"
        ].replace(0, 0.1)

    # Packet size variations
    if "Packet Length Std" in X.columns and "Packet Length Mean" in X.columns:
        X["Packet_Size_CV"] = X["Packet Length Std"] / X["Packet Length Mean"].replace(
            0, 0.1
        )

    return X


def create_flag_features(X):
    """
    Create features based on TCP flags
    """
    flag_cols = [col for col in X.columns if "Flag" in col]
    if len(flag_cols) > 1:
        # Total flags
        X["Total_Flags"] = X[flag_cols].sum(axis=1)

        # Flag diversity (number of different flags used)
        X["Flag_Diversity"] = (X[flag_cols] > 0).sum(axis=1)

        # Ratio of control flags (SYN, FIN, RST) to total flags
        control_flags = ["SYN Flag Count", "FIN Flag Count", "RST Flag Count"]
        if all(flag in X.columns for flag in control_flags):
            X["Control_Flag_Ratio"] = X[control_flags].sum(axis=1) / X[
                "Total_Flags"
            ].replace(0, 1)

    return X


def feature_engineering(X):
    """
    Comprehensive feature engineering combining multiple aspects
    """
    X = X.copy()

    # Convert string features to numeric
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Create features from different aspects
    X = create_timing_features(X)
    X = create_packet_features(X)
    X = create_flag_features(X)

    # Fill any NaN values created during feature engineering
    X.fillna(0, inplace=True)

    return X


def select_features(X, y, threshold=0.01):
    """
    Perform feature selection using multiple methods
    """
    RF = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    RF.fit(X, y)
    importances = RF.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Option 1: Select features based on importance threshold
    selected_indices = [
        i for i, importance in enumerate(importances) if importance >= threshold
    ]
    selected_features = [RF.feature_names_[i] for i in selected_indices]

    # Option 2 (alternative): Select top N features
    # If you want to select a specific number of top features instead, uncomment this:
    # num_features = int(len(feature_names) * threshold) if threshold < 1 else int(threshold)
    # selected_features = [feature_names[i] for i in indices[:num_features]]

    return selected_features, importances


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model with metrics prioritizing recall for DoS detection
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


# Main execution flow
if __name__ == "__main__":
    # Load and analyze data
    print("Loading and analyzing training data...")
    train = load_data("data/train/train.csv")
    test = load_data("data/test/test.csv")

    X_train = train.drop("Label", axis=1)
    y_train = train["Label"]
    X_test = test.drop("Label", axis=1)
    y_test = test["Label"]

    # Feature engineering
    print("\nPerforming feature engineering...")
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    # Analyze features
    print("\nAnalyzing features...")
    low_variance, high_corr = analyze_features(X_train)
    # remove near constant features
    print(f"Low variance features to drop: {low_variance}")
    X_train = X_train.drop(low_variance, axis=1)
    X_test = X_test.drop(low_variance, axis=1)
    # remove highly correlated features
    high_corr_feat_to_drop = []
    for feat1, feat2, corr in high_corr:
        if corr > 0.95:
            if feat1 not in high_corr_feat_to_drop and feat1 not in low_variance:
                high_corr_feat_to_drop.append(feat1)

    print(f"Highly correlated features to drop: {high_corr_feat_to_drop}")
    X_train = X_train.drop(high_corr_feat_to_drop, axis=1)
    X_test = X_test.drop(high_corr_feat_to_drop, axis=1)

    # Feature selection
    print("\nPerforming feature selection...")
    selected_features, importances = select_features(X_train, y_train, threshold=0.01)
    print(f"Selected features: {selected_features}")
    print(f"Importances: {importances}")

    # Train model
    print("\nTraining model...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train[selected_features], y_train)
    evaluate_model(model, X_test[selected_features], y_test)

    pickle.dump(model, open("model/rf.pkl", "wb"))
