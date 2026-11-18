from pathlib import Path

import numpy as np
import pandas as pd


class CICIDSPreprocessor:
    TARGET_COLUMN = "Label"
    BENIGN_LABEL = "BENIGN"
    EXPECTED_FEATURE_COUNT = 78

    def __init__(
        self,
        csv_path: str | Path,
        positive_labels: tuple[str, ...] = ("DDoS",),
    ) -> None:
        if not positive_labels:
            raise ValueError("Debe indicar al menos una etiqueta positiva")
        if self.BENIGN_LABEL in positive_labels:
            raise ValueError(
                f"La etiqueta '{self.BENIGN_LABEL}' está reservada para la clase 0"
            )
        self.csv_path = Path(csv_path)
        self.positive_labels = tuple(dict.fromkeys(positive_labels))
        self.label_mapping = {self.BENIGN_LABEL: 0}
        self.label_mapping.update({label: 1 for label in self.positive_labels})

    def load_dataset(self) -> pd.DataFrame:
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el dataset en la ruta indicada: {self.csv_path}"
            )
        try:
            return pd.read_csv(self.csv_path)
        except (
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            UnicodeDecodeError,
            OSError,
        ) as error:
            raise ValueError(
                f"No se pudo leer el dataset '{self.csv_path}': {error}"
            ) from error

    def clean_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        cleaned = dataset.copy()
        cleaned.columns = cleaned.columns.str.strip()
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
        cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)
        return cleaned

    def build_features_and_target(
        self, dataset: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        if self.TARGET_COLUMN not in dataset.columns:
            raise KeyError(
                f"La columna objetivo '{self.TARGET_COLUMN}' no existe en el dataset"
            )

        labels = dataset[self.TARGET_COLUMN]
        unsupported_labels = sorted(set(labels.unique()) - set(self.label_mapping))
        if unsupported_labels:
            raise ValueError(
                f"Etiquetas no soportadas en '{self.TARGET_COLUMN}': "
                f"{unsupported_labels}. Etiquetas configuradas: "
                f"{sorted(self.label_mapping)}"
            )

        features = dataset.drop(columns=[self.TARGET_COLUMN])
        if len(features.columns) != self.EXPECTED_FEATURE_COUNT:
            raise ValueError(
                "Se esperaban "
                f"{self.EXPECTED_FEATURE_COUNT} características y se encontraron "
                f"{len(features.columns)}"
            )
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
            raise TypeError(
                "Todas las características deben ser numéricas para entrenar el modelo"
            )

        target = labels.map(self.label_mapping).astype(int)
        return features, target

    def process(self) -> tuple[pd.DataFrame, pd.Series]:
        dataset = self.load_dataset()
        cleaned_dataset = self.clean_dataset(dataset)
        return self.build_features_and_target(cleaned_dataset)
