# 🧬 Synthetic Data Sandbox (VAE Edition)

Piattaforma end-to-end per la generazione di **dataset tabellari sintetici anonimi** tramite **Variational Autoencoder (VAE)** in PyTorch, validazione statistica/privacy e interrogazione in linguaggio naturale tramite **LLM Code Generation** (esecuzione sandboxed di codice Pandas/Matplotlib).

## 🏗️ Architettura del Sistema

1. **Data Engine (`backend/data_engine.py`)**: Imputazione dei valori mancanti, normalizzazione (`MinMaxScaler`), codifica delle variabili categoriche (`OneHotEncoder`) e ricostruzione inversa completa (`inverse_transform`) con ripristino dei tipi interi e delle categorie originali.
2. **Generative Core (`backend/vae_model.py`)**: Rete neurale Variational Autoencoder in PyTorch con *Reparameterization Trick*, `LayerNorm`, attivazione finale `Sigmoid` e *KL Annealing* ($\beta$-VAE) per evitare il posterior collapse e preservare varianza e correlazioni.
3. **Validation & Privacy Engine (`backend/validator.py`)**:
   - Confronto statistico (media, deviazione standard, min, max).
   - Matrici di correlazione affiancate e punteggio di similarità strutturale.
   - Confronto delle distribuzioni tramite istogrammi sovrapposti.
   - **Privacy Check**: verifica assenza di duplicati esatti (*Exact Matches*) e calcolo della *Distance to Closest Record (DCR)* media.
4. **Natural Language Analyst (`backend/llm_analyst.py`)**: Integrazione tramite OpenAI SDK (compatibile sia con **Ollama** in locale che con **OpenAI API** cloud) per tradurre domande in codice Python ed eseguirlo in un ambiente isolato con whitelist di moduli.
5. **Backend API (`backend/main.py`) & UI (`frontend/app.py`)**: Servizio REST in **FastAPI** (`/upload`, `/train`, `/generate`, `/validate`, `/analyze`) collegato a una dashboard interattiva **Streamlit**.

---

## 🚀 Avvio Rapido

### Opzione A: Avvio con Docker (Consigliato)

Avvia contemporaneamente backend FastAPI e frontend Streamlit già configurati:

```bash
docker compose up --build