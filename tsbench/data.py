"""Download and load UCR/UEA datasets from timeseriesclassification.com."""
import io
import os
import urllib.request
import zipfile

import numpy as np

URL = "https://www.timeseriesclassification.com/aeon-toolkit/{name}.zip"
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download(name, root=DEFAULT_DIR):
    folder = os.path.join(root, name)
    if os.path.exists(os.path.join(folder, f"{name}_TRAIN.txt")):
        return folder
    os.makedirs(folder, exist_ok=True)
    with urllib.request.urlopen(URL.format(name=name), timeout=60) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(folder)
    return folder


def _read(path):
    arr = np.loadtxt(path, ndmin=2)
    return arr[:, 1:], arr[:, 0]


def load(name, root=DEFAULT_DIR):
    """Return X_train, y_train, X_test, y_test. X has shape (n, length).

    Labels are mapped to 0..k-1 in sorted order of the original labels.
    """
    folder = download(name, root)
    Xtr, ytr = _read(os.path.join(folder, f"{name}_TRAIN.txt"))
    Xte, yte = _read(os.path.join(folder, f"{name}_TEST.txt"))
    classes = np.unique(np.concatenate([ytr, yte]))
    lut = {c: i for i, c in enumerate(classes)}
    ytr = np.array([lut[c] for c in ytr])
    yte = np.array([lut[c] for c in yte])
    return Xtr, ytr, Xte, yte
