"""A minimal ROCKET-style classifier: random convolutional kernels + ridge regression.

Follows the idea of Dempster et al. (2020): many random dilated kernels, two features per
kernel (max and proportion of positive values), then a linear classifier. This is a compact
re-implementation for learning and benchmarking, not the reference package.
"""
import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler


def _make_kernels(n_kernels, length, rng):
    lengths = rng.choice([7, 9, 11], n_kernels)
    kernels, biases, dilations, paddings = [], [], [], []
    for l in lengths:
        w = rng.normal(0, 1, l)
        w -= w.mean()
        max_exp = np.log2(max((length - 1) / (l - 1), 1.0))
        d = int(2 ** rng.uniform(0, max_exp))
        kernels.append(w)
        biases.append(rng.uniform(-1, 1))
        dilations.append(d)
        paddings.append(((l - 1) * d) // 2 if rng.random() < 0.5 else 0)
    return kernels, np.array(biases), np.array(dilations), np.array(paddings)


@njit(cache=True)
def _apply(X, w, b, dilation, padding):
    n, length = X.shape
    k = w.shape[0]
    out_len = length + 2 * padding - (k - 1) * dilation
    feats = np.zeros((n, 2))
    for i in range(n):
        best = -np.inf
        pos = 0
        for s in range(-padding, -padding + out_len):
            v = b
            for j in range(k):
                idx = s + j * dilation
                if 0 <= idx < length:
                    v += w[j] * X[i, idx]
            if v > best:
                best = v
            if v > 0:
                pos += 1
        feats[i, 0] = pos / out_len
        feats[i, 1] = best
    return feats


class MiniRocket(BaseEstimator, ClassifierMixin):
    def __init__(self, n_kernels=500, random_state=0):
        self.n_kernels = n_kernels
        self.random_state = random_state

    def _transform(self, X):
        X = np.ascontiguousarray(X, dtype=np.float64)
        cols = [
            _apply(X, np.asarray(w), b, d, p)
            for w, b, d, p in zip(self.kernels_, self.biases_, self.dilations_, self.paddings_)
        ]
        return np.hstack(cols)

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        self.kernels_, self.biases_, self.dilations_, self.paddings_ = _make_kernels(
            self.n_kernels, X.shape[1], rng
        )
        F = self._transform(X)
        self.scaler_ = StandardScaler().fit(F)
        self.clf_ = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(self.scaler_.transform(F), y)
        self.classes_ = self.clf_.classes_
        return self

    def predict(self, X):
        return self.clf_.predict(self.scaler_.transform(self._transform(X)))
