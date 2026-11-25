"""Run the benchmark: python -m tsbench.run"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from .data import load
from .rocket import MiniRocket

DATASETS = ["ItalyPowerDemand", "GunPoint", "ECG200", "Coffee", "CBF"]


def classifiers():
    return {
        "1NN-Euclidean": lambda: KNeighborsTimeSeriesClassifier(1, metric="euclidean"),
        "1NN-DTW (full)": lambda: KNeighborsTimeSeriesClassifier(1, metric="dtw"),
        "1NN-DTW (band 10%)": lambda: KNeighborsTimeSeriesClassifier(
            1, metric="dtw", metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": None}
        ),
        "MiniRocket-500": lambda: MiniRocket(500, random_state=0),
    }


def _band_radius(length, frac=0.10):
    return max(1, int(round(frac * length)))


def classify(name):
    Xtr, ytr, Xte, yte = load(name)
    scaler = TimeSeriesScalerMeanVariance()
    Xtr3 = scaler.fit_transform(Xtr)
    Xte3 = scaler.transform(Xte)
    rows = []
    for label, make in classifiers().items():
        clf = make()
        if "band" in label:
            clf.set_params(metric_params={"global_constraint": "sakoe_chiba",
                                          "sakoe_chiba_radius": _band_radius(Xtr.shape[1])})
        t0 = time.time()
        if label.startswith("MiniRocket"):
            clf.fit(Xtr3[:, :, 0], ytr)
            pred = clf.predict(Xte3[:, :, 0])
        else:
            clf.fit(Xtr3, ytr)
            pred = clf.predict(Xte3)
        rows.append(dict(dataset=name, method=label, accuracy=accuracy_score(yte, pred),
                         seconds=time.time() - t0, n_train=len(ytr), n_test=len(yte), length=Xtr.shape[1]))
    return rows


def cluster(name, seeds=(0, 1, 2, 3, 4)):
    Xtr, ytr, Xte, yte = load(name)
    X = TimeSeriesScalerMeanVariance().fit_transform(np.vstack([Xtr, Xte]))
    y = np.concatenate([ytr, yte])
    k = len(np.unique(y))
    rows = []
    for metric in ["euclidean", "dtw"]:
        scores = []
        t0 = time.time()
        for s in seeds:
            km = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=15, n_init=1,
                                  max_iter_barycenter=10, random_state=s)
            scores.append(adjusted_rand_score(y, km.fit_predict(X)))
        rows.append(dict(dataset=name, method=f"KMeans-{metric}", ari_mean=float(np.mean(scores)),
                         ari_std=float(np.std(scores)), seeds=len(seeds), seconds=time.time() - t0))
    return rows


def main():
    clf_rows, clu_rows = [], []
    for name in DATASETS:
        print("classify", name, flush=True)
        clf_rows += classify(name)
    for name in ["ItalyPowerDemand", "GunPoint", "ECG200", "Coffee"]:
        print("cluster", name, flush=True)
        clu_rows += cluster(name)
    clf = pd.DataFrame(clf_rows)
    clu = pd.DataFrame(clu_rows)
    clf.to_csv("results/classification.csv", index=False)
    clu.to_csv("results/clustering.csv", index=False)
    print(clf.pivot(index="dataset", columns="method", values="accuracy").round(3).to_string())
    print(clu.pivot(index="dataset", columns="method", values="ari_mean").round(3).to_string())


if __name__ == "__main__":
    main()
