"""Task definitions shared by models and evaluation (heads, losses and metrics are chosen per task)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    key: str
    label: str
    output_dim: str            # "1" or "num_classes"
    loss: str
    primary_metrics: tuple


TASK_SPECS = {
    "binary_classification": TaskSpec("binary_classification", "Binary classification", "1", "BCEWithLogitsLoss",
                                      ("pr_auc", "roc_auc", "precision", "recall", "f1", "confusion_matrix")),
    "multiclass_classification": TaskSpec("multiclass_classification", "Multiclass classification", "num_classes",
                                          "CrossEntropyLoss", ("macro_f1", "weighted_f1", "accuracy", "per_class", "confusion_matrix")),
    "regression": TaskSpec("regression", "Regression", "1", "MSELoss", ("mae", "mse", "rmse", "r2")),
}
