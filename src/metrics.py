from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ModelMetrics:
    precision: float
    recall: float
    f1_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    def to_dict(self) -> dict[str, object]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": {
                "tn": self.true_negatives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
                "tp": self.true_positives,
            },
        }


def calculate_metrics(target: pd.Series, predictions: pd.Series) -> ModelMetrics:
    true_negatives, false_positives, false_negatives, true_positives = (
        confusion_matrix(target, predictions, labels=[0, 1]).ravel()
    )
    return ModelMetrics(
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1_score=float(f1_score(target, predictions, zero_division=0)),
        true_negatives=int(true_negatives),
        false_positives=int(false_positives),
        false_negatives=int(false_negatives),
        true_positives=int(true_positives),
    )
