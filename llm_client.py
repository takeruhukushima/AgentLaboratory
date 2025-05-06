"""
llm_client.py
共通 LLM 呼び出しラッパー
--------------------------------------
backend 例:
    - "o3-mini"        → OpenAI (oモデル群)
    - "gemini-2.0-flash" / "gemini-2.5-pro"
"""

import os, functools
from dotenv import load_dotenv
load_dotenv()

# ---- モデル名エイリアス ------------------------------
MODEL_ALIASES = {
    "o3-mini"      : "gpt-4o-mini",
    "o4-mini-high" : "gpt-4o-mini",
    "gpt-4o-mini"  : "gpt-4o-mini",
    # Gemini 側
    "gemini-flash" : "gemini-2.0-flash",
    "gemini-pro"   : "gemini-2.5-pro",
}

def canonical(name: str) -> str:
    return MODEL_ALIASES.get(name, name)

# -----------------------------------------------------

class LLMClient:
    """OpenAI / Gemini を透過的に使う小さな Facade"""
    def __init__(self, backend: str):
        self.backend = backend.lower()
        if self.backend.startswith("gemini"):
            from google import genai
            key = os.getenv("GOOGLE_API_KEY")
            if not key:
                raise EnvironmentError("GOOGLE_API_KEY missing in environment/.env")
            self.genai = genai.Client(api_key=key)
        else:  # OpenAI 系
            import openai
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise EnvironmentError("OPENAI_API_KEY missing in environment/.env")
            openai.api_key = key
            self.openai = openai

    # -------- パブリックメソッド ----------------------
    def chat(self, prompt: str, model: str, temperature: float = 0.7, **kw) -> str:
        model = canonical(model)

        if model.startswith("gemini"):
            resp = self.genai.models.generate_content(
                model=model,
                contents=prompt,
                generation_config={
                    "temperature": temperature,
                    **kw,
                },
            )
            return resp.text

        # OpenAI fallback
        resp = self.openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **kw,
        )
        return resp["choices"][0]["message"]["content"]

# -----------------------------------------------------
# シングルトンキャッシュ
_client_cache = {}
def get_client(model: str) -> "LLMClient":
    if model not in _client_cache:
        _client_cache[model] = LLMClient(model)
    return _client_cache[model]
