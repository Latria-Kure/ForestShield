import subprocess
import pandas as pd
import numpy as np
from forest_shield.forest import RandomForestClassifier
import time
import os
import signal
import pickle

os.system("tput reset")
capture_cmd_template = "sudo tcpdump -i ens33 -w {} > /dev/null 2>&1"
analyze_cmd_template = "cfm {} capture > /dev/null 2>&1"
pcap_file_template = "capture/capture{}.pcap"
out_csv_template = "capture/capture{}.pcap_Flow.csv"
ALL_LABELS = ["BENIGN", "DoS Hulk", "DoS GoldenEye", "DoS slowloris"]
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


def load_data(file_path):
    data = pd.read_csv(file_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.drop(REMOVE_FEATURES, axis=1, inplace=True)
    data.drop("Label", axis=1, inplace=True)
    return data


def get_selected_features(model: RandomForestClassifier):
    return model.feature_names_


def capture(capture_file_num: int):
    pcap_file = pcap_file_template.format(capture_file_num)
    capture_cmd = capture_cmd_template.format(pcap_file)
    capture_process = subprocess.Popen(
        ["sudo", "tcpdump", "-i", "ens33", "-w", pcap_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return capture_process


def analyze(capture_file_num: int):
    pcap_file = pcap_file_template.format(capture_file_num)
    analyze_cmd = analyze_cmd_template.format(pcap_file)
    analyze_process = subprocess.Popen(analyze_cmd, shell=True)
    return analyze_process


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


def dos_detect(classifier: RandomForestClassifier, period: int = 3):
    """
    Detect DoS attacks in real-time using a trained classifier.

    Args:
        classifier: Trained RandomForestClassifier
        period: Number of seconds to capture and analyze data
    """
    # check if capture dir exists
    selected_features = get_selected_features(classifier)
    print(f"Selected features: {selected_features}")

    if not os.path.exists("capture"):
        os.makedirs("capture")

    # Remove existing pcap files if they exist
    for capture_file_num in range(1, 3):
        pcap_file = pcap_file_template.format(capture_file_num)
        if os.path.exists(pcap_file):
            os.remove(pcap_file)

    # capture first file
    print("Preparing...\r")
    capture_file_num = 1
    capture_process = capture(capture_file_num)
    time.sleep(period)
    os.kill(capture_process.pid, signal.SIGINT)

    analyze_file_num = 1
    analyze_process = analyze(analyze_file_num)

    capture_file_num = 2
    capture_process = capture(capture_file_num)

    curr_time = time.time()
    result = {label: 0 for label in ALL_LABELS}
    print("Start detecting...\r")
    while True:
        analyze_finished = analyze_process.poll() is not None
        capture_finished = capture_process.poll() is not None

        if analyze_finished and capture_finished:
            data = load_data(out_csv_template.format(analyze_file_num))
            data = feature_engineering(data)
            data = data[selected_features]
            label = classifier.predict(data)
            for i in range(len(label)):
                result[label[i]] += 1

            # Print with newline and carriage return
            print(f"{result}\r", flush=True)

            result = {label: 0 for label in ALL_LABELS}
            analyze_file_num = 1 if analyze_file_num == 2 else 2
            capture_file_num = 1 if capture_file_num == 2 else 2
            analyze_process = analyze(analyze_file_num)
            capture_process = capture(capture_file_num)
            curr_time = time.time()
        else:
            if time.time() - curr_time > period and not capture_finished:
                # Ensure proper line formatting for status messages
                try:
                    # Try SIGTERM if SIGINT doesn't work
                    os.kill(capture_process.pid, signal.SIGTERM)
                    # Give it a moment to terminate
                    time.sleep(0.2)
                    # Check if it's still running
                    if capture_process.poll() is None:
                        # Try SIGKILL as a last resort
                        os.kill(capture_process.pid, signal.SIGKILL)
                except OSError as e:
                    print(f"Error killing process: {e}\r", flush=True)
                continue


if __name__ == "__main__":
    classifier = pickle.load(open("model/rf.pkl", "rb"))
    dos_detect(classifier)
