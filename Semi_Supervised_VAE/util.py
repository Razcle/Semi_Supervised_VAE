import gzip
import os

import numpy as np

from sklearn.model_selection import train_test_split


def load_mnist(n_labelled=100):
    def load_images(fn):
        with gzip.open(fn, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28 * 28)
        return data / np.float32(255)

    def load_labels(fn):
        with gzip.open(fn, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    file_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(file_dir, "..", "data")

    fn_train_images = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    X_train = load_images(fn_train_images)

    fn_train_labels = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    y_train = load_labels(fn_train_labels)

    fn_test_images = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    X_test = load_images(fn_test_images)

    fn_test_labels = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    y_test = load_labels(fn_test_labels)

    X_labelled, X_unlabelled, y, _ = train_test_split(X_train, y_train,
                                                      train_size=n_labelled,
                                                      stratify=y_train)

    return X_unlabelled, X_labelled, y, X_test, y_test
