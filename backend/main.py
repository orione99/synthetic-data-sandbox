import io
import os
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.data_engine import TabularDataEngine
from backend.llm_analyst import NaturalLanguageAnalyst
from backend.vae_model import VAETrainer
from backend.validator import SyntheticDataValidator

app = FastAPI(title="Synthetic Data Sandbox API (VAE Edition)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)


# Stato in memoria della sessione corrente
class SandboxState:
    df_real: pd.DataFrame | None = None
    df_synth: pd.DataFrame | None = None
    data_engine: TabularDataEngine | None = None
    trainer: VAETrainer | None = None
    training_history: list[dict] = []
    filename: str | None = None


state = SandboxState()
validator = SyntheticDataValidator()
analyst = NaturalLanguageAnalyst()


class TrainRequest(BaseModel):
    epochs: int = 150
    latent_dim: int = 8
    batch_size: int = 32
    lr: float = 1e-3
    max_beta: float = 0.015  # Valore bilanciato per preservare varianza e correlazioni


class AnalyzeRequest(BaseModel):
    question: str
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "dataset_loaded": state.df_real is not None,
        "model_trained": state.trainer is not None,
        "synthetic_ready": state.df_synth is not None,
    }


@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Carica un file in formato .csv")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Impossibile leggere il CSV: {e}")

    if df.empty or df.shape[1] < 2:
        raise HTTPException(status_code=400, detail="Il CSV deve contenere almeno 2 colonne e 1 riga.")

    state.df_real = df
    state.df_synth = None
    state.trainer = None
    state.data_engine = None
    state.training_history = []
    state.filename = file.filename

    df.to_csv("data/uploaded_real.csv", index=False)

    return {
        "message": "Dataset caricato con successo",
        "filename": file.filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "preview": df.head(5).fillna("").to_dict(orient="records"),
    }


@app.post("/train")
def train_vae(req: TrainRequest):
    if state.df_real is None:
        raise HTTPException(status_code=400, detail="Nessun dataset caricato. Chiama prima /upload.")

    engine = TabularDataEngine()
    X_scaled = engine.fit_transform(state.df_real)

    trainer = VAETrainer(
        input_dim=X_scaled.shape[1],
        latent_dim=req.latent_dim,
        lr=req.lr,
    )
    history = trainer.train(
        X_scaled,
        epochs=req.epochs,
        batch_size=req.batch_size,
        max_beta=req.max_beta,
    )

    state.data_engine = engine
    state.trainer = trainer
    state.training_history = history

    # Genera subito un primo dataset sintetico della stessa dimensione dell'originale
    raw_synth = trainer.generate(num_samples=len(state.df_real))
    state.df_synth = engine.inverse_transform(raw_synth)
    state.df_synth.to_csv("data/generated_synthetic.csv", index=False)

    return {
        "message": "Addestramento VAE completato",
        "device": str(trainer.device),
        "final_metrics": history[-1],
        "history": history,
    }


@app.get("/generate")
def generate_synthetic(num_samples: int | None = None, download: bool = True):
    if state.trainer is None or state.data_engine is None or state.df_real is None:
        raise HTTPException(status_code=400, detail="Modello non ancora addestrato. Chiama prima /train.")

    n = num_samples if (num_samples and num_samples > 0) else len(state.df_real)
    raw_synth = state.trainer.generate(num_samples=n)
    df_synth = state.data_engine.inverse_transform(raw_synth)

    state.df_synth = df_synth
    df_synth.to_csv("data/generated_synthetic.csv", index=False)

    if download:
        stream = io.StringIO()
        df_synth.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=synthetic_dataset.csv"
        return response

    return {
        "rows": int(df_synth.shape[0]),
        "preview": df_synth.head(10).fillna("").to_dict(orient="records"),
    }


@app.get("/validate")
def validate_synthetic():
    if state.df_real is None or state.df_synth is None:
        raise HTTPException(status_code=400, detail="Dati reali o sintetici mancanti.")

    summary_df = validator.summary_comparison(state.df_real, state.df_synth)
    corr_data = validator.correlation_comparison(state.df_real, state.df_synth)
    dist_b64 = validator.distribution_comparison(state.df_real, state.df_synth)
    privacy_data = validator.privacy_check(state.df_real, state.df_synth)

    return {
        "summary_table": summary_df.to_dict(orient="records"),
        "correlation": corr_data,
        "distributions_base64": dist_b64,
        "privacy": privacy_data,
    }


@app.post("/analyze")
def analyze_with_llm(req: AnalyzeRequest):
    if state.df_synth is None:
        raise HTTPException(status_code=400, detail="Genera prima il dataset sintetico per interrogarlo.")

    result = analyst.ask(
        question=req.question,
        df_synth=state.df_synth,
        base_url=req.base_url,
        api_key=req.api_key,
        model=req.model,
    )
    return result