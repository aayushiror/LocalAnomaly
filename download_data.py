"""
Download and prepare NSL-KDD dataset for network anomaly detection.
Dataset source: Canadian Institute for Cybersecurity
URL: https://www.unb.ca/cic/datasets/nsl.html
Also available on GitHub: https://github.com/jmnwong/NSL-KDD-Dataset
"""

import os
import urllib.request
import pandas as pd
import numpy as np

#NSLKDD
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

TRAIN_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"
TEST_URL  = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved to {dest_path}")


def load_raw(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    return df


def preprocess(df):
    """Binary label: normal=0, attack=1"""
    df = df.copy()
    df["binary_label"] = (df["label"] != "normal").astype(int)
    df["attack_type"] = df["label"].apply(lambda x: x if x == "normal" else x.split(".")[0])
    df.drop(columns=["label", "difficulty"], inplace=True)
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    train_raw = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_raw  = os.path.join(DATA_DIR, "KDDTest+.txt")

    download_file(TRAIN_URL, train_raw)
    download_file(TEST_URL,  test_raw)

    print("\nLoading and preprocessing...")
    train_df = preprocess(load_raw(train_raw))
    test_df  = preprocess(load_raw(test_raw))

    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_df.to_csv( os.path.join(DATA_DIR, "test.csv"),  index=False)

    print(f"\nTrain shape : {train_df.shape}")
    print(f"Test  shape : {test_df.shape}")
    print(f"\nLabel distribution (train):\n{train_df['binary_label'].value_counts()}")
    print(f"\nAttack types (train):\n{train_df['attack_type'].value_counts()}")
    print("\nData ready in data/train.csv and data/test.csv")


if __name__ == "__main__":
    main()
