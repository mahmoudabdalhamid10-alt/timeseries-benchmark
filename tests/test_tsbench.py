import numpy as np
from sklearn.metrics import accuracy_score

from tsbench.rocket import MiniRocket


def _toy(n=60, length=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, length)
    X0 = np.sin(t) + 0.3 * rng.normal(size=(n, length))
    X1 = np.sign(np.sin(t)) + 0.3 * rng.normal(size=(n, length))
    return np.vstack([X0, X1]), np.array([0] * n + [1] * n)


def test_rocket_separates_toy_classes():
    X, y = _toy()
    Xt, yt = _toy(seed=1)
    clf = MiniRocket(200, random_state=0).fit(X, y)
    assert accuracy_score(yt, clf.predict(Xt)) > 0.9


def test_rocket_is_deterministic():
    X, y = _toy()
    a = MiniRocket(100, random_state=3).fit(X, y).predict(X)
    b = MiniRocket(100, random_state=3).fit(X, y).predict(X)
    assert (a == b).all()


def test_loader_label_mapping(tmp_path):
    from tsbench.data import load
    d = tmp_path / "Toy"
    d.mkdir()
    np.savetxt(d / "Toy_TRAIN.txt", np.array([[1, 0.1, 0.2], [2, 0.3, 0.4]]))
    np.savetxt(d / "Toy_TEST.txt", np.array([[2, 0.5, 0.6]]))
    Xtr, ytr, Xte, yte = load("Toy", root=str(tmp_path))
    assert Xtr.shape == (2, 2) and list(ytr) == [0, 1] and list(yte) == [1]
