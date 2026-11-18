import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.artifacts import validate_feature_order
from src.metrics import ModelMetrics, calculate_metrics
from src.preprocessing import CICIDSPreprocessor


class UnsupervisedAnomalyTrainer:
    MODEL_FILENAME = "iforest_anomaly_model.joblib"
    METRICS_FILENAME = "iforest_metrics.json"

    def __init__(
        self,
        ddos_path: str | Path,
        portscan_path: str | Path,
        models_dir: str | Path,
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 200,
        contamination: float = 0.05,
    ) -> None:
        self.ddos_path = Path(ddos_path)
        self.portscan_path = Path(portscan_path)
        self.models_dir = Path(models_dir)
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.contamination = contamination

    def run(self) -> dict[str, object]:
        features, target = CICIDSPreprocessor(
            self.ddos_path, positive_labels=("DDoS",)
        ).process()
        features_train, features_test, target_train, target_test = self._split(
            features, target
        )

        benign_features = features_train[target_train == 0]
        scaler = StandardScaler().fit(benign_features)
        model = self._build_model()
        model.fit(scaler.transform(benign_features))

        print("Entrenamiento no supervisado con Isolation Forest")
        print(f"Filas de entrenamiento totales: {len(features_train)}")
        print(f"Filas BENIGN usadas para ajustar el modelo: {len(benign_features)}")
        print(f"Contaminación configurada: {self.contamination}")

        ddos_metrics = self._evaluate(
            "DDoS", model, scaler, features_test, target_test
        )
        reports: dict[str, object] = {
            "configuracion": {
                "contaminacion": self.contamination,
                "n_estimators": self.n_estimators,
                "test_size": self.test_size,
                "random_state": self.random_state,
                "filas_entrenamiento": int(len(features_train)),
                "filas_benign": int(len(benign_features)),
            },
            "ddos": self._build_report("DDoS", target_test, ddos_metrics),
        }

        portscan_report = self._evaluate_portscan(
            model, scaler, features.columns.tolist()
        )
        if portscan_report is not None:
            reports["portscan"] = portscan_report

        self._persist(model, scaler, features.columns.tolist(), reports)
        return reports

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

    def _build_model(self) -> IsolationForest:
        return IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _predict(
        self,
        model: IsolationForest,
        scaler: StandardScaler,
        features: pd.DataFrame,
    ) -> pd.Series:
        raw_predictions = model.predict(scaler.transform(features))
        return pd.Series(
            np.where(raw_predictions == -1, 1, 0), index=features.index
        )

    def _evaluate(
        self,
        dataset_name: str,
        model: IsolationForest,
        scaler: StandardScaler,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> ModelMetrics:
        predictions = self._predict(model, scaler, features)
        metrics = calculate_metrics(target, predictions)

        print(f"\nEvaluación no supervisada sobre {dataset_name}:")
        print(f"  Filas evaluadas: {len(target)}")
        print(f"  Distribución de clases: {self._format_distribution(dataset_name, target)}")
        print(f"  Precisión: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1-score: {metrics.f1_score:.4f}")
        print("  Matriz de confusión:")
        print(
            f"    TN: {metrics.true_negatives} | FP: {metrics.false_positives} | "
            f"FN: {metrics.false_negatives} | TP: {metrics.true_positives}"
        )
        return metrics

    def _evaluate_portscan(
        self,
        model: IsolationForest,
        scaler: StandardScaler,
        feature_names: list[str],
    ) -> dict[str, object] | None:
        if not self.portscan_path.is_file():
            print(
                "\nNo se encontró el dataset PortScan en "
                f"'{self.portscan_path}'; se omite su evaluación."
            )
            return None
        features, target = CICIDSPreprocessor(
            self.portscan_path, positive_labels=("PortScan",)
        ).process()
        validate_feature_order(feature_names, features.columns)
        metrics = self._evaluate("PortScan", model, scaler, features, target)
        return self._build_report("PortScan", target, metrics)

    def _format_distribution(self, dataset_name: str, target: pd.Series) -> str:
        distribution = target.value_counts().sort_index()
        return ", ".join(
            f"{'BENIGN' if label == 0 else dataset_name} ({label}): {count}"
            for label, count in distribution.items()
        )

    def _build_report(
        self, dataset_name: str, target: pd.Series, metrics: ModelMetrics
    ) -> dict[str, object]:
        distribution = {
            f"{'BENIGN' if label == 0 else dataset_name} ({label})": int(count)
            for label, count in target.value_counts().sort_index().items()
        }
        return {
            "dataset": dataset_name,
            "filas_evaluadas": int(len(target)),
            "distribucion_clases": distribution,
            **metrics.to_dict(),
        }

    def _persist(
        self,
        model: IsolationForest,
        scaler: StandardScaler,
        feature_names: list[str],
        reports: dict[str, object],
    ) -> None:
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            model_path = self.models_dir / self.MODEL_FILENAME
            metrics_path = self.models_dir / self.METRICS_FILENAME
            joblib.dump(
                {
                    "scaler": scaler,
                    "model": model,
                    "feature_names": feature_names,
                    "metrics": reports,
                },
                model_path,
            )
            metrics_path.write_text(
                json.dumps(reports, indent=2, ensure_ascii=False),
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
    raw_data_dir = project_root / "data" / "raw"
    trainer = UnsupervisedAnomalyTrainer(
        ddos_path=raw_data_dir
        / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        portscan_path=raw_data_dir
        / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        models_dir=project_root / "models",
    )
    trainer.run()


if __name__ == "__main__":
    main()
