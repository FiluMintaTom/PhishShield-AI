"""SHAP explanations for the saved URL RandomForest model."""
import logging
import os
from functools import lru_cache

import joblib
import numpy as np
import shap

LOG = logging.getLogger(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "url_model.joblib")
FEATURE_PATH = os.path.join(BASE, "models", "feature_names.joblib")

@lru_cache(maxsize=1)
def _load():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Trained URL model not found: {MODEL_PATH}")
    if not os.path.isfile(FEATURE_PATH):
        raise FileNotFoundError(f"Feature-name file not found: {FEATURE_PATH}")
    model = joblib.load(MODEL_PATH)
    names = list(joblib.load(FEATURE_PATH))
    if len(names) != getattr(model, "n_features_in_", len(names)):
        raise ValueError(f"Feature count mismatch: model expects {model.n_features_in_}, feature file has {len(names)}")
    return model, names, shap.TreeExplainer(model)

def explain(feature_dict, top_k=8):
    model, names, explainer = _load()
    missing = [name for name in names if name not in feature_dict]
    if missing:
        raise ValueError(f"URL feature vector is missing trained features: {missing}")
    X = np.asarray([[feature_dict[name] for name in names]], dtype=float)
    if X.shape[1] != model.n_features_in_:
        raise ValueError(f"SHAP input has {X.shape[1]} features; model expects {model.n_features_in_}")
    raw = explainer.shap_values(X, check_additivity=False)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError(f"Phishing label 1 not present in model classes: {classes}")
    class_index = classes.index(1)
    if isinstance(raw, list):
        if class_index >= len(raw):
            raise ValueError(f"SHAP returned {len(raw)} class arrays; requested class index {class_index}")
        vals = np.asarray(raw[class_index], dtype=float).reshape(-1, len(names))[0]
    else:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 3:  # SHAP >= 0.45 commonly: samples, features, classes
            if arr.shape[0] == 1 and arr.shape[1] == len(names) and arr.shape[2] == len(classes):
                vals = arr[0, :, class_index]
            elif arr.shape[0] == len(classes) and arr.shape[1] == 1 and arr.shape[2] == len(names):
                vals = arr[class_index, 0, :]
            else:
                raise ValueError(f"Unrecognized 3-D SHAP output shape {arr.shape}")
        elif arr.ndim == 2 and arr.shape == (1, len(names)):
            # Binary single-output explainers explain model output; accept only if output is unambiguous.
            vals = arr[0]
        else:
            raise ValueError(f"Unrecognized SHAP output shape {arr.shape}")
    if len(vals) != len(names) or not np.all(np.isfinite(vals)):
        raise ValueError("SHAP returned invalid or non-finite feature contributions")
    ranked = sorted(zip(names, vals), key=lambda item: abs(float(item[1])), reverse=True)
    return [{"feature": n, "value": feature_dict[n], "shap": round(float(v), 6),
             "direction": "increases phishing score" if v > 0 else "reduces phishing score" if v < 0 else "no contribution"}
            for n, v in ranked[:top_k]]
