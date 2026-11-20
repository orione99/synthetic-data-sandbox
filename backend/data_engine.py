import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class TabularDataEngine:
    """
    Gestisce preprocessing (imputazione nulli, MinMaxScaler, OneHotEncoder)
    e ricostruzione inversa (inverse_transform) per dati tabellari.
    """

    def __init__(self):
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.int_cols: list[str] = []
        self.original_columns: list[str] = []

        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.original_columns = list(df.columns)

        # Identifica colonne numeriche e categoriche
        self.num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        self.cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
        self.int_cols = df.select_dtypes(include=["int64", "int32"]).columns.tolist()

        transformed_parts = []

        if self.num_cols:
            num_data = self.num_imputer.fit_transform(df[self.num_cols])
            num_scaled = self.scaler.fit_transform(num_data)
            transformed_parts.append(num_scaled)

        if self.cat_cols:
            cat_data = self.cat_imputer.fit_transform(df[self.cat_cols].astype(str))
            cat_encoded = self.encoder.fit_transform(cat_data)
            transformed_parts.append(cat_encoded)

        self.is_fitted = True
        return np.hstack(transformed_parts).astype(np.float32)

    def inverse_transform(self, synthetic_array: np.ndarray) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Il DataEngine deve essere addestrato prima di chiamare inverse_transform.")

        reconstructed_parts = {}
        n_num = len(self.num_cols)

        # 1. Ricostruzione variabili numeriche
        if n_num > 0:
            num_scaled = np.clip(synthetic_array[:, :n_num], 0.0, 1.0)
            num_original = self.scaler.inverse_transform(num_scaled)

            for idx, col in enumerate(self.num_cols):
                values = num_original[:, idx]
                if col in self.int_cols:
                    values = np.round(values).astype(int)
                reconstructed_parts[col] = values

        # 2. Ricostruzione variabili categoriche tramite argmax per ciascun blocco One-Hot
        if self.cat_cols:
            cat_encoded = synthetic_array[:, n_num:]
            offset = 0
            for idx, col in enumerate(self.cat_cols):
                categories = self.encoder.categories_[idx]
                n_cats = len(categories)
                block = cat_encoded[:, offset : offset + n_cats]
                chosen_indices = np.argmax(block, axis=1)
                reconstructed_parts[col] = categories[chosen_indices]
                offset += n_cats

        # Ripristina l'ordine originale delle colonne
        df_synthetic = pd.DataFrame(reconstructed_parts)[self.original_columns]
        return df_synthetic