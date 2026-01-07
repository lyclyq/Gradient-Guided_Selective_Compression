import numpy as np
from sklearn.metrics import accuracy_score

def compute_acc(preds, labels):
    return float(accuracy_score(labels, preds))

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)
