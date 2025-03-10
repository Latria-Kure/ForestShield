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


def dos_detect(classifier: RandomForestClassifier, period: int = 3):
    """
    Detect DoS attacks in real-time using a trained classifier.

    Args:
        classifier: Trained RandomForestClassifier
        period: Number of seconds to capture and analyze data
    """
    # check if capture dir exists
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
