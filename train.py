"""
Minimal training script for MLForge E2E test.
Trains an SGDClassifier on the Iris dataset and saves with joblib.
This file exists in the user's repo and gets cloned by env_manager.
"""
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def train_and_save(output_dir=".", epochs=6, random_state=42):
    """Train an SGD classifier and save it."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    classes = np.unique(y)
    clf = SGDClassifier(loss="log_loss", random_state=random_state)
    for epoch in range(1, epochs + 1):
        clf.partial_fit(X_train, y_train, classes=classes)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "sgd_iris.joblib")
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path} | accuracy={acc:.4f}")
    return model_path, acc
if __name__ == "__main__":
    train_and_save()
