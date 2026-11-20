import base64
import io
import matplotlib
matplotlib.use("Agg")  # Backend non interattivo sicuro per FastAPI/server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


class SyntheticDataValidator:
    """
    Valuta la fedeltà statistica e la privacy dei dati sintetici rispetto ai reali.
    """

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def summary_comparison(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> pd.DataFrame:
        num_cols = df_real.select_dtypes(include=["number"]).columns
        rows = []
        for col in num_cols:
            rows.append({
                "Variabile": col,
                "Media Reale": round(float(df_real[col].mean()), 2),
                "Media Sintetica": round(float(df_synth[col].mean()), 2),
                "Std Reale": round(float(df_real[col].std()), 2),
                "Std Sintetica": round(float(df_synth[col].std()), 2),
                "Min Reale": round(float(df_real[col].min()), 2),
                "Min Sintetico": round(float(df_synth[col].min()), 2),
                "Max Reale": round(float(df_real[col].max()), 2),
                "Max Sintetico": round(float(df_synth[col].max()), 2),
            })
        return pd.DataFrame(rows)

    def correlation_comparison(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
        num_cols = df_real.select_dtypes(include=["number"]).columns
        corr_real = df_real[num_cols].corr().fillna(0)
        corr_synth = df_synth[num_cols].corr().fillna(0)

        # Differenza media assoluta tra le matrici di correlazione (0 = identiche, 2 = opposte)
        mae_corr = float(np.abs(corr_real.values - corr_synth.values).mean())
        similarity_score = round(max(0.0, (1.0 - mae_corr) * 100), 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(corr_real, ax=axes[0], cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", cbar=False)
        axes[0].set_title("Correlazione Dati Reali")

        sns.heatmap(corr_synth, ax=axes[1], cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f")
        axes[1].set_title("Correlazione Dati Sintetici")

        plt.tight_layout()
        img_b64 = self._fig_to_base64(fig)

        return {
            "correlation_similarity_pct": similarity_score,
            "heatmap_base64": img_b64,
        }

    def distribution_comparison(
        self, df_real: pd.DataFrame, df_synth: pd.DataFrame, columns: list[str] | None = None
    ) -> str:
        num_cols = df_real.select_dtypes(include=["number"]).columns.tolist()
        selected_cols = columns if columns else num_cols[:6]  # Max 6 grafici in griglia 2x3

        n_cols = len(selected_cols)
        rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for idx, col in enumerate(selected_cols):
            ax = axes[idx]
            sns.histplot(df_real[col], ax=ax, color="#1f77b4", label="Reale", stat="density", bins=20, alpha=0.45)
            sns.histplot(df_synth[col], ax=ax, color="#ff7f0e", label="Sintetico", stat="density", bins=20, alpha=0.45)
            ax.set_title(f"Distribuzione: {col}")
            ax.legend()

        # Nasconde eventuali subplot vuoti
        for idx in range(n_cols, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return self._fig_to_base64(fig)

    def privacy_check(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> dict:
        # 1. Exact matches (righe identiche al 100%)
        common_cols = list(df_real.columns)
        merged = pd.merge(
            df_real[common_cols].drop_duplicates(),
            df_synth[common_cols],
            on=common_cols,
            how="inner",
        )
        exact_matches = int(len(merged))

        # 2. Distance to Closest Record (DCR) sulle variabili numeriche scalate in [0, 1]
        num_cols = df_real.select_dtypes(include=["number"]).columns
        scaler = MinMaxScaler()
        real_scaled = scaler.fit_transform(df_real[num_cols].fillna(0))
        synth_scaled = scaler.transform(df_synth[num_cols].fillna(0))

        # Calcola distanza euclidea minima di ciascun record sintetico dal record reale più vicino
        dist_matrix = cdist(synth_scaled, real_scaled, metric="euclidean")
        min_distances = dist_matrix.min(axis=1)

        return {
            "exact_matches": exact_matches,
            "exact_match_rate_pct": round((exact_matches / max(1, len(df_synth))) * 100, 2),
            "mean_dcr": round(float(min_distances.mean()), 4),
            "min_dcr": round(float(min_distances.min()), 4),
            "privacy_passed": exact_matches == 0 and float(min_distances.mean()) > 0.05,
        }


# Test rapido da terminale
if __name__ == "__main__":
    from data_engine import TabularDataEngine
    from vae_model import VAETrainer

    df_real = pd.read_csv("data/diabetes.csv")
    engine = TabularDataEngine()
    X_scaled = engine.fit_transform(df_real)

    trainer = VAETrainer(input_dim=X_scaled.shape[1], latent_dim=8)
    trainer.train(X_scaled, epochs=150, batch_size=32)
    df_synth = engine.inverse_transform(trainer.generate(len(df_real)))

    validator = SyntheticDataValidator()
    print("=== TABELLA COMPARATIVA ===")
    print(validator.summary_comparison(df_real, df_synth).to_string(index=False))

    corr_res = validator.correlation_comparison(df_real, df_synth)
    print(f"\nSimilarità Correlazioni: {corr_res['correlation_similarity_pct']}%")

    priv_res = validator.privacy_check(df_real, df_synth)
    print("\n=== PRIVACY CHECK ===")
    print(priv_res)