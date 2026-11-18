import json
import sys
from pathlib import Path

import pandas as pd

from src.artifacts import load_model_artifact, validate_feature_order
from src.metrics import ModelMetrics, calculate_metrics
from src.preprocessing import CICIDSPreprocessor


class GeneralizationEvaluator:
    METRICS_FILENAME = "generalization_metrics.json"

    def __init__(
        self,
        model_path: str | Path,
        dataset_path: str | Path,
        models_dir: str | Path,
        positive_labels: tuple[str, ...] = ("PortScan",),
        supervised_metrics_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.models_dir = Path(models_dir)
        self.positive_labels = positive_labels
        self.supervised_metrics_path = (
            Path(supervised_metrics_path)
            if supervised_metrics_path is not None
            else self.models_dir / "rf_metrics.json"
        )

    def run(self) -> dict[str, object]:
        artifact = load_model_artifact(self.model_path)
        model = artifact["model"]
        feature_names = artifact["feature_names"]

        features, target = CICIDSPreprocessor(
            self.dataset_path, positive_labels=self.positive_labels
        ).process()
        validate_feature_order(feature_names, features.columns)

        print(f"Evaluación de generalización sobre: {self.dataset_path.name}")
        print(f"Etiquetas positivas: {', '.join(self.positive_labels)}")
        print(f"Filas evaluadas: {len(target)}")

        predictions = pd.Series(model.predict(features), index=features.index)
        metrics = calculate_metrics(target, predictions)
        self._print_distribution(target)
        self._print_metrics(metrics)

        report = self._build_report(target, metrics)
        baseline_recall = self._load_supervised_recall()
        if baseline_recall is not None:
            report["recall_ddos_supervisado"] = baseline_recall
            drop = (baseline_recall - metrics.recall) * 100
            print(
                f"\nRecall supervisado en DDoS: {baseline_recall:.4f} | "
                f"Caída de Recall frente a ataques no vistos: {drop:.2f} puntos porcentuales"
            )

        self._persist(report)
        return report

    def _print_distribution(self, target: pd.Series) -> None:
        distribution = target.value_counts().sort_index()
        detail = ", ".join(
            f"{self._class_name(label)} ({label}): {count}"
            for label, count in distribution.items()
        )
        print(f"Distribución de clases: {detail}")

    def _print_metrics(self, metrics: ModelMetrics) -> None:
        print("\nMétricas de generalización:")
        print(f"  Precisión: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1-score: {metrics.f1_score:.4f}")
        print("Matriz de confusión:")
        print(
            f"  TN: {metrics.true_negatives} | FP: {metrics.false_positives} | "
            f"FN: {metrics.false_negatives} | TP: {metrics.true_positives}"
        )

    def _build_report(
        self, target: pd.Series, metrics: ModelMetrics
    ) -> dict[str, object]:
        distribution = {
            f"{self._class_name(label)} ({label})": int(count)
            for label, count in target.value_counts().sort_index().items()
        }
        return {
            "dataset": self.dataset_path.name,
            "etiquetas_positivas": list(self.positive_labels),
            "filas_evaluadas": int(len(target)),
            "distribucion_clases": distribution,
            **metrics.to_dict(),
        }

    def _class_name(self, label: int) -> str:
        if label == 0:
            return "BENIGN"
        return " / ".join(self.positive_labels)

    def _load_supervised_recall(self) -> float | None:
        if not self.supervised_metrics_path.is_file():
            print(
                "\nNo se encontró la referencia supervisada "
                f"'{self.supervised_metrics_path}'; se omite la comparación de Recall."
            )
            return None
        try:
            supervised_metrics = json.loads(
                self.supervised_metrics_path.read_text(encoding="utf-8")
            )
            return float(supervised_metrics["recall"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(
                "\nNo se pudo leer la referencia supervisada "
                f"'{self.supervised_metrics_path}': {error}. "
                "Se omite la comparación de Recall."
            )
            return None

    def _persist(self, report: dict[str, object]) -> None:
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = self.models_dir / self.METRICS_FILENAME
            metrics_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as error:
            raise RuntimeError(
                f"No se pudieron guardar las métricas en '{self.models_dir}': {error}"
            ) from error
        print(f"\nMétricas guardadas en: {metrics_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    evaluator = GeneralizationEvaluator(
        model_path=project_root / "models" / "rf_ddos_model.joblib",
        dataset_path=raw_data_dir
        / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        models_dir=project_root / "models",
        positive_labels=("PortScan",),
    )
    evaluator.run()


if __name__ == "__main__":
    main()
