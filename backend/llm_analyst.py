import ast
import base64
import contextlib
import io
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import seaborn as sns


class NaturalLanguageAnalyst:
    """
    Traduce domande in linguaggio naturale in codice Python (Pandas/Matplotlib)
    tramite LLM (OpenAI SDK -> Ollama locale o OpenAI Cloud) e lo esegue in modo sicuro.
    """

    FORBIDDEN_TOKENS = (
        "import os",
        "import sys",
        "import subprocess",
        "import shutil",
        "__import__",
        "open(",
        "eval(",
        "exec(",
        "compile(",
        "globals(",
        "locals(",
    )

    ALLOWED_MODULES = {
        "pandas",
        "numpy",
        "matplotlib",
        "matplotlib.pyplot",
        "seaborn",
        "math",
        "statistics",
    }

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b")

        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)

    def _build_system_prompt(self, df: pd.DataFrame) -> str:
        schema_lines = [f"- {col} ({dtype})" for col, dtype in df.dtypes.items()]
        schema_str = "\n".join(schema_lines)
        sample_str = df.head(3).to_string(index=False)

        return (
            "Sei un Data Analyst esperto di Python. Hai a disposizione un DataFrame pandas già caricato "
            "nella variabile `df` (contiene dati sintetici anonimi).\n\n"
            f"STRUTTURA DEL DATAFRAME (`df`, {len(df)} righe):\n{schema_str}\n\n"
            f"ESEMPIO PRIME 3 RIGHE:\n{sample_str}\n\n"
            "REGOLE TASSATIVE:\n"
            "1. Rispondi SOLO con un singolo blocco di codice Python racchiuso tra ```python e ```.\n"
            "2. Usa esclusivamente le librerie già importate: `df`, `pd`, `np`, `plt`, `sns`.\n"
            "3. NON ricaricare CSV, NON usare `plt.show()` e NON usare `os`, `sys` o file I/O.\n"
            "4. Stampa sempre con `print(...)` i risultati numerici chiave o una breve spiegazione testuale.\n"
            "5. Se la domanda beneficia di una visualizzazione (es. correlazioni, distribuzioni, confronti), "
            "crea un grafico chiaro con `plt.figure(figsize=(8, 5))`, titolo ed etichette sugli assi."
        )

    @staticmethod
    def _extract_code(raw_response: str) -> str:
        match = re.search(r"```(?:python)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return raw_response.strip()

    def _validate_code_safety(self, code: str) -> None:
        lowered = code.lower()
        for token in self.FORBIDDEN_TOKENS:
            if token in lowered:
                raise ValueError(f"Codice bloccato per sicurezza: rilevato comando non consentito ('{token}').")
        # Verifica sintattica prima dell'esecuzione
        ast.parse(code)

    @classmethod
    def _safe_import(cls, name, globals=None, locals=None, fromlist=(), level=0):
        if name in cls.ALLOWED_MODULES or name.split(".")[0] in {"matplotlib"}:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Modulo non consentito nella sandbox: '{name}'")

    def ask(
        self,
        question: str,
        df_synth: pd.DataFrame,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> dict:
        active_base_url = base_url or self.base_url
        active_api_key = api_key or self.api_key
        active_model = model or self.model

        client = openai.OpenAI(base_url=active_base_url, api_key=active_api_key)
        system_prompt = self._build_system_prompt(df_synth)

        response = client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content or ""
        code = self._extract_code(raw_content)

        try:
            self._validate_code_safety(code)
        except Exception as e:
            return {
                "question": question,
                "generated_code": code,
                "text_output": f"Errore di validazione codice: {e}",
                "plot_base64": None,
                "success": False,
            }

        # Evita che eventuali plt.show() generati dall'LLM blocchino o puliscano la figura
        cleaned_code = re.sub(r"plt\.show\s*\(\s*\)", "", code)

        plt.close("all")
        stdout_buffer = io.StringIO()
        safe_globals = {
            "__builtins__": {
                "__import__": self._safe_import,
                "print": print,
                "len": len,
                "range": range,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "float": float,
                "int": int,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "zip": zip,
                "enumerate": enumerate,
            },
            "df": df_synth.copy(),
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
        }

        plot_b64 = None
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(cleaned_code, safe_globals, {})

            # Se il codice ha generato una figura Matplotlib, la esporta in base64
            if plt.get_fignums():
                fig = plt.gcf()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
                buf.seek(0)
                plot_b64 = base64.b64encode(buf.read()).decode("utf-8")
                plt.close("all")

            text_out = stdout_buffer.getvalue().strip()
            if not text_out and plot_b64:
                text_out = "Grafico generato con successo."
            elif not text_out:
                text_out = "Analisi eseguita senza output testuale."

            return {
                "question": question,
                "generated_code": code,
                "text_output": text_out,
                "plot_base64": plot_b64,
                "success": True,
            }

        except Exception as e:
            plt.close("all")
            return {
                "question": question,
                "generated_code": code,
                "text_output": f"Errore durante l'esecuzione del codice: {type(e).__name__}: {e}",
                "plot_base64": None,
                "success": False,
            }