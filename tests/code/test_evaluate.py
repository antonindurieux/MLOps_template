import numpy as np

from tagifai import evaluate


def test_get_metrics():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    classes = ["a", "b"]
    performance = evaluate.get_metrics(y_true=y_true, y_pred=y_pred, classes=classes, df=None)
    assert performance["overall"]["precision"] == 2 / 4
    assert performance["overall"]["recall"] == 2 / 4
    assert performance["class"]["a"]["precision"] == 1 / 2
    assert performance["class"]["a"]["recall"] == 1 / 2
    assert performance["class"]["b"]["precision"] == 1 / 2
    assert performance["class"]["b"]["recall"] == 1 / 2
