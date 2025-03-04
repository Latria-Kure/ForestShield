import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    RobustScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
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
    near_constant = variance[variance < 0.01].index.tolist()
    print(f"\nNear-constant features (variance < 0.01): {near_constant}")

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

    return near_constant, high_corr_pairs


def load_data(file_path):
    """
    Load and perform initial data cleaning
    """
    print(f"\nLoading data from {file_path}")
    data = pd.read_csv(file_path, nrows=10000)

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

    # Flow rate features
    if all(feat in X.columns for feat in ["Flow Bytes/s", "Flow Packets/s"]):
        X["Bytes_per_Packet"] = X["Flow Bytes/s"] / X["Flow Packets/s"].replace(0, 0.1)

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


def select_features(X, y, feature_names):
    """
    Perform feature selection using multiple methods
    """
    # 1. Remove low variance features
    selector1 = VarianceThreshold(threshold=0.01)
    X_var = selector1.fit_transform(X)
    selected_mask = selector1.get_support()

    # Get selected feature names
    selected_features = [
        f for f, selected in zip(feature_names, selected_mask) if selected
    ]
    print(
        f"\nFeatures removed by variance threshold: {len(feature_names) - len(selected_features)}"
    )

    # 2. Use Random Forest for feature importance
    rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=50, random_state=42), prefit=False
    )
    rf_selector.fit(X_var, y)

    # Get final selected features
    final_mask = rf_selector.get_support()
    final_features = [
        f for f, selected in zip(selected_features, final_mask) if selected
    ]

    print("\nFinal selected features:")
    for f in final_features:
        print(f"- {f}")

    return rf_selector, final_features


def preprocess_data(data):
    """
    Preprocess the data with enhanced feature handling
    """
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(data["Label"])

    # Store class mapping
    class_mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"\nClass Mapping: {class_mapping}")

    # Separate features
    X = data.drop(["Label"], axis=1)
    feature_names = X.columns.tolist()

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Create preprocessing pipelines
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),  # RobustScaler handles outliers better
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    )

    return X, y, preprocessor, le, feature_names


def train_model(X_train, y_train, preprocessor):
    """
    Train a baseline Random Forest model
    """
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the model with metrics prioritizing recall for DoS detection
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Get the original class names
    class_names = label_encoder.classes_

    # Classification metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations to the confusion matrix
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(conf_matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if conf_matrix[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("confusion_matrix.png")

    # Key metrics for DoS detection (multi-class)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    f2 = fbeta_score(y_test, y_pred, beta=2, average="macro")

    print(f"\nKey Metrics (Macro-averaged):")
    print(f"Recall (sensitivity): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score (emphasizes recall): {f2:.4f}")

    # Return metrics for potential further analysis
    return {"recall": recall, "precision": precision, "f1": f1, "f2": f2}


def plot_feature_importance(model, X_train, top_n=20):
    """
    Fixed feature importance function that works directly with the trained model
    """
    # Get feature importances from the classifier
    rf_model = model.named_steps["classifier"]
    feature_importances = rf_model.feature_importances_

    # Create a simpler list of feature names - fixes the error in previous code
    # Instead of trying to recover exact transformed feature names,
    # we'll create generic feature names based on the originals
    feature_names = X_train.columns.tolist()

    # Sort features by importance
    indices = np.argsort(feature_importances)[::-1]

    # Plot top_n most important features
    plt.figure(figsize=(12, 8))
    n_features = min(top_n, len(feature_importances))

    # Create the plot
    plt.title("Feature Importances for DoS Detection")
    plt.bar(
        range(n_features), feature_importances[indices[:n_features]], align="center"
    )
    plt.xticks(
        range(n_features), [feature_names[i] for i in indices[:n_features]], rotation=90
    )
    plt.tight_layout()
    plt.savefig("feature_importance.png")

    # Print the top features
    print("\nTop Features by Importance:")
    for i in range(n_features):
        print(
            f"{i+1}. {feature_names[indices[i]]}: {feature_importances[indices[i]]:.4f}"
        )


# Main execution flow
if __name__ == "__main__":
    # Load and analyze data
    print("Loading and analyzing training data...")
    train = load_data("data/train/train_capture_hulk.csv")
    test = load_data("data/test/test.csv")

    # Analyze features
    print("\nAnalyzing features...")
    near_constant, high_corr = analyze_features(train.drop("Label", axis=1))

    # Preprocess data
    print("\nPreprocessing data...")
    X_train, y_train, preprocessor, label_encoder, feature_names = preprocess_data(
        train
    )
    X_test, y_test, _, _, _ = preprocess_data(test)

    # Feature engineering
    print("\nPerforming feature engineering...")
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    # Feature selection
    print("\nPerforming feature selection...")
    selector, selected_features = select_features(X_train, y_train, X_train.columns)

    # Train model
    print("\nTraining baseline model...")
    model = train_model(X_train, y_train, preprocessor)

    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, label_encoder)

    # Plot feature importance
    plot_feature_importance(model, X_train)

    # Save artifacts
    joblib.dump(model, "dos_detection_model.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("\nModel and label encoder saved successfully")
