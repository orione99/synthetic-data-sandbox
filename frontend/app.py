import base64
import io
import os
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DEFAULT_OLLAMA_URL = os.getenv("DEFAULT_OLLAMA_URL", "http://localhost:11434/v1")

st.set_page_config(
    page_title="Synthetic Data Sandbox (VAE Edition)",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Synthetic Data Sandbox (VAE Edition)")
st.caption("Generazione di dataset tabellari anonimi tramite Variational Autoencoder e analisi in linguaggio naturale.")

# Inizializza stato sessione Streamlit
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_info" not in st.session_state:
    st.session_state.upload_info = None
if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "validation_data" not in st.session_state:
    st.session_state.validation_data = None
if "synth_preview" not in st.session_state:
    st.session_state.synth_preview = None


def decode_b64_image(b64_str: str) -> bytes:
    return base64.b64decode(b64_str)


# --- SIDEBAR: Controlli Pipeline ---
with st.sidebar:
    st.header("1. Carica Dataset CSV")
    uploaded_file = st.file_uploader("Trascina qui il tuo file .csv", type=["csv"])

    if uploaded_file is not None:
        if st.button("Invia al Backend", use_container_width=True):
            with st.spinner("Caricamento dataset..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.session_state.upload_info = res.json()
                    st.session_state.train_result = None
                    st.session_state.validation_data = None
                    st.session_state.synth_preview = None
                    st.session_state.chat_history = []
                    st.success(f"Caricato: {st.session_state.upload_info['rows']} righe")
                else:
                    st.error(res.json().get("detail", "Errore durante l'upload"))

    st.divider()
    st.header("2. Addestramento VAE")
    epochs = st.slider("Epoche", min_value=50, max_value=400, value=150, step=25)
    latent_dim = st.slider("Dimensione Spazio Latente", min_value=2, max_value=32, value=8, step=2)
    max_beta = st.select_slider(
        "Peso KL (Beta)",
        options=[0.005, 0.015, 0.03, 0.05, 0.1],
        value=0.015,
        help="Valori bassi (es. 0.015) preservano meglio varianza e correlazioni su dataset piccoli.",
    )

    train_disabled = st.session_state.upload_info is None
    if st.button("Addestra VAE & Genera", type="primary", disabled=train_disabled, use_container_width=True):
        with st.spinner("Addestramento rete neurale in corso..."):
            payload = {"epochs": epochs, "latent_dim": latent_dim, "batch_size": 32, "max_beta": max_beta}
            train_res = requests.post(f"{API_URL}/train", json=payload)

            if train_res.status_code == 200:
                st.session_state.train_result = train_res.json()

                # Recupera anteprima sintetici e metriche di validazione
                prev_res = requests.get(f"{API_URL}/generate", params={"download": False})
                if prev_res.status_code == 200:
                    st.session_state.synth_preview = prev_res.json()["preview"]

                val_res = requests.get(f"{API_URL}/validate")
                if val_res.status_code == 200:
                    st.session_state.validation_data = val_res.json()

                st.success(f"Training completato su {st.session_state.train_result['device'].upper()}!")
            else:
                st.error(train_res.json().get("detail", "Errore durante l'addestramento"))

    st.divider()
    st.header("3. Configurazione LLM & Export")
    provider = st.radio("Provider LLM", ["Ollama (Locale)", "OpenAI API"], horizontal=True)

    if provider == "Ollama (Locale)":
        llm_base_url = DEFAULT_OLLAMA_URL
        llm_api_key = "ollama"
        llm_model = st.text_input("Modello Ollama", value="qwen2.5:7b")
    else:
        llm_base_url = "https://api.openai.com/v1"
        llm_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        llm_model = st.text_input("Modello OpenAI", value="gpt-4o-mini")

    if st.session_state.train_result is not None:
        num_gen = st.number_input("Righe da scaricare", min_value=10, max_value=10000, value=768, step=100)
        dl_res = requests.get(f"{API_URL}/generate", params={"num_samples": num_gen, "download": True})
        if dl_res.status_code == 200:
            st.download_button(
                label="⬇️ Scarica CSV Sintetico",
                data=dl_res.content,
                file_name="synthetic_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

# --- CORPO PRINCIPALE: Tab ---
if st.session_state.upload_info is None:
    st.info("👈 Carica un file CSV dalla barra laterale per iniziare (es. `data/diabetes.csv`).")
else:
    tab1, tab2, tab3 = st.tabs([
        "📊 Dataset & Training VAE",
        "🧪 Validazione Statistica & Privacy",
        "🤖 Natural Language Analyst",
    ])

    # TAB 1: Dati Reali vs Sintetici + Loss Curve
    with tab1:
        info = st.session_state.upload_info
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("File", info["filename"])
        col_m2.metric("Righe Originali", info["rows"])
        col_m3.metric("Colonne", info["columns"])

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Anteprima Dati Reali")
            st.dataframe(pd.DataFrame(info["preview"]), use_container_width=True)

        with c2:
            st.subheader("Anteprima Dati Sintetici")
            if st.session_state.synth_preview:
                st.dataframe(pd.DataFrame(st.session_state.synth_preview), use_container_width=True)
            else:
                st.warning("Avvia l'addestramento del VAE dalla sidebar per generare i dati sintetici.")

        if st.session_state.train_result:
            st.subheader("Andamento Loss durante l'Addestramento")
            hist_df = pd.DataFrame(st.session_state.train_result["history"]).set_index("epoch")
            st.line_chart(hist_df[["total_loss", "recon_loss", "kl_loss"]])

    # TAB 2: Validazione & Privacy
    with tab2:
        val = st.session_state.validation_data
        if not val:
            st.info("Addestra il VAE per visualizzare le metriche di qualità e privacy.")
        else:
            priv = val["privacy"]
            corr = val["correlation"]

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Similarità Correlazioni", f"{corr['correlation_similarity_pct']}%")
            p2.metric("Record Identici (Exact Match)", priv["exact_matches"])
            p3.metric("Distanza Media (Mean DCR)", priv["mean_dcr"])
            p4.metric("Privacy Check", "✅ SUPERATO" if priv["privacy_passed"] else "⚠️ ATTENZIONE")

            st.subheader("Tabella Comparativa: Dati Reali vs Sintetici")
            st.dataframe(pd.DataFrame(val["summary_table"]), use_container_width=True, hide_index=True)

            st.subheader("Confronto Matrici di Correlazione")
            st.image(decode_b64_image(corr["heatmap_base64"]), use_container_width=True)

            st.subheader("Confronto Distribuzioni delle Variabili")
            st.image(decode_b64_image(val["distributions_base64"]), use_container_width=True)

    # TAB 3: Chatbot Analyst
    with tab3:
        if not st.session_state.synth_preview:
            st.info("Genera prima il dataset sintetico per interrogarlo con l'LLM.")
        else:
            st.markdown("Fai una domanda sui **dati sintetici**. L'LLM genererà ed eseguirà codice Python per risponderti.")
            st.caption("Esempi: *'Qual è la correlazione tra Glucose e BMI?'* oppure *'Mostra la distribuzione dell'età divisa per Outcome con un grafico'*")

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("plot"):
                        st.image(decode_b64_image(msg["plot"]))
                    if msg.get("code"):
                        with st.expander("Codice Python generato ed eseguito"):
                            st.code(msg["code"], language="python")

            user_question = st.chat_input("Chiedi qualcosa al dataset sintetico...")
            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("L'LLM sta scrivendo ed eseguendo l'analisi..."):
                        ans_res = requests.post(
                            f"{API_URL}/analyze",
                            json={
                                "question": user_question,
                                "model": llm_model,
                                "base_url": llm_base_url,
                                "api_key": llm_api_key,
                            },
                        )
                        if ans_res.status_code == 200:
                            data = ans_res.json()
                            st.markdown(data["text_output"])
                            if data.get("plot_base64"):
                                st.image(decode_b64_image(data["plot_base64"]))
                            with st.expander("Codice Python generato ed eseguito"):
                                st.code(data["generated_code"], language="python")

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": data["text_output"],
                                "plot": data.get("plot_base64"),
                                "code": data["generated_code"],
                            })
                        else:
                            err_msg = ans_res.json().get("detail", "Errore nella chiamata all'LLM")
                            st.error(err_msg)