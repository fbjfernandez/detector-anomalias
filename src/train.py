import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.metrics import ModelMetrics, calculate_metrics
from src.preprocessing import CICIDSPreprocessor


class SupervisedTrainer:
    MODEL_FILENAME = "rf_ddos_model.joblib"
    METRICS_FILENAME = "rf_metrics.json"
    CLASS_NAMES = {0: "BENIGN", 1: "DDoS"}

    def __init__(
        self,
        dataset_path: str | Path,
        models_dir: str | Path = "models",
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 200,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.models_dir = Path(models_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators

    def run(self) -> ModelMetrics:
        features, target = CICIDSPreprocessor(self.dataset_path).process()
        self._print_deduplicated_rows(len(features))

        features_train, features_test, target_train, target_test = self._split(
            features, target
        )
        self._print_class_distribution("entrenamiento", target_train)
        self._print_class_distribution("prueba", target_test)

        model = self._build_model()
        model.fit(features_train, target_train)
        predictions = model.predict(features_test)

        metrics = self._calculate_metrics(target_test, predictions)
        self._print_metrics(metrics)

        feature_names = features.columns.tolist()
        self._print_top_features(model, feature_names)
        self._persist_artifacts(model, feature_names, metrics)
        return metrics

    def _split(
        self, features: pd.DataFrame, target: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            return train_test_split(
                features,
                target,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=target,
            )
        except ValueError as error:
            raise ValueError(
                f"No se pudo dividir el dataset en entrenamiento y prueba: {error}"
            ) from error

    def _build_model(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _calculate_metrics(
        self, target_test: pd.Series, predictions: pd.Series
    ) -> ModelMetrics:
        return calculate_metrics(target_test, predictions)

    def _print_deduplicated_rows(self, total_rows: int) -> None:
        print(f"Filas tras deduplicación: {total_rows}")

    def _print_class_distribution(self, dataset_name: str, target: pd.Series) -> None:
        distribution = target.value_counts().sort_index()
        detail = ", ".join(
            f"{self.CLASS_NAMES[label]} ({label}): {count}"
            for label, count in distribution.items()
        )
        print(f"Distribución de clases en {dataset_name}: {detail}")

    def _print_metrics(self, metrics: ModelMetrics) -> None:
        print("\nMétricas en el conjunto de prueba:")
        print(f"  Precisión: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1-score: {metrics.f1_score:.4f}")
        print("Matriz de confusión:")
        print(
            f"  TN: {metrics.true_negatives} | FP: {metrics.false_positives} | "
            f"FN: {metrics.false_negatives} | TP: {metrics.true_positives}"
        )

    def _print_top_features(
        self, model: RandomForestClassifier, feature_names: list[str]
    ) -> None:
        ranked_features = sorted(
            zip(feature_names, model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
        print("\nTop 10 de importancia de características:")
        for position, (name, importance) in enumerate(ranked_features[:10], start=1):
            print(f"  {position:2d}. {name}: {importance:.4f}")

    def _persist_artifacts(
        self,
        model: RandomForestClassifier,
        feature_names: list[str],
        metrics: ModelMetrics,
    ) -> None:
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.models_dir / self.MODEL_FILENAME
            metrics_path = self.models_dir / self.METRICS_FILENAME
            joblib.dump(
                {
                    "model": model,
                    "feature_names": feature_names,
                    "metrics": metrics.to_dict(),
                },
                model_path,
            )
            metrics_path.write_text(
                json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as error:
            raise RuntimeError(
                f"No se pudieron guardar los artefactos en '{self.models_dir}': {error}"
            ) from error
        print(f"\nModelo guardado en: {model_path}")
        print(f"Métricas guardadas en: {metrics_path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    project_root = Path(__file__).resolve().parent.parent
    trainer = SupervisedTrainer(
        dataset_path=project_root
        / "data"
        / "raw"
        / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        models_dir=project_root / "models",
    )
    trainer.run()


if __name__ == "__main__":
    main()
