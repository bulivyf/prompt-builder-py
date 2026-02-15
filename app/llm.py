from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import httpx


@dataclass
class LLMConfig:
    provider: str  # local|openai|anthropic|google|grok|hf
    model: str
    temperature: float = 0.2


def resolve_provider() -> str:
    """Provider selection is **explicit**.

    - Default is LOCAL, even if external API keys exist.
    - Remote providers are only used if LLM_PROVIDER is set to a non-local value.
    """
    provider = (os.getenv("LLM_PROVIDER") or "local").strip().lower()
    if provider in {"auto", ""}:
        # Backward compatible: treat "auto" as local-first.
        return "local"
    return provider


def get_default_model(provider: str) -> str:
    if provider == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    if provider == "google":
        return os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    if provider == "grok":
        return os.getenv("GROK_MODEL", "grok-2")
    if provider == "hf":
        return os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    return os.getenv("LOCAL_MODEL_NAME", "local-gguf")


class LLMClient:
    def __init__(self) -> None:
        self.provider = resolve_provider()
        self.model = get_default_model(self.provider)
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    async def complete(self, system: str, user: str) -> str:
        # Local-first default. Remote only if explicitly chosen.
        if self.provider == "local":
            return await self._local_complete(system, user)
        if self.provider == "openai":
            return await self._openai_compatible_complete(
                system,
                user,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        if self.provider == "grok":
            return await self._openai_compatible_complete(
                system,
                user,
                api_key=os.getenv("GROK_API_KEY"),
                base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
            )
        if self.provider == "anthropic":
            return await self._anthropic_complete(system, user)
        if self.provider == "google":
            return await self._google_complete(system, user)
        if self.provider == "hf":
            return await self._hf_inference_complete(system, user)

        # Unknown provider -> fail safe to local
        return await self._local_complete(system, user)

    async def _openai_compatible_complete(self, system: str, user: str, api_key: Optional[str], base_url: str) -> str:
        if not api_key:
            raise RuntimeError("Missing API key for OpenAI-compatible provider (set LLM_PROVIDER=local to use local).")
        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]

    async def _anthropic_complete(self, system: str, user: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY (or set LLM_PROVIDER=local).")
        try:
            from anthropic import AsyncAnthropic  # type: ignore

            client = AsyncAnthropic(api_key=api_key)
            msg = await client.messages.create(
                model=self.model,
                max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "1200")),
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join([b.text for b in msg.content if hasattr(b, "text")])
        except Exception:
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.model,
                "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1200")),
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return "".join([blk.get("text", "") for blk in data.get("content", [])])

    async def _google_complete(self, system: str, user: str) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or set LLM_PROVIDER=local).")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(user, generation_config={"temperature": self.temperature})
        return getattr(resp, "text", "") or ""

    async def _hf_inference_complete(self, system: str, user: str) -> str:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Missing HF_TOKEN (or set LLM_PROVIDER=local).")
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {token}"}
        prompt = f"""{system}\n\nUser:\n{user}\n\nAssistant:\n"""
        payload = {"inputs": prompt, "parameters": {"temperature": self.temperature, "max_new_tokens": 1200}}
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data and "generated_text" in data[0]:
                return data[0]["generated_text"][len(prompt):].strip()
            if isinstance(data, dict) and "generated_text" in data:
                return str(data["generated_text"])
            return json.dumps(data)

    async def _local_complete(self, system: str, user: str) -> str:
        """Local GGUF via llama-cpp-python."""
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/model.gguf")
        n_ctx = int(os.getenv("LOCAL_MODEL_N_CTX", "4096"))
        n_threads = int(os.getenv("LOCAL_MODEL_N_THREADS", "8"))

        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Local provider selected but llama-cpp-python is not available. "
                "Install it (pip install llama-cpp-python) and set LOCAL_MODEL_PATH to a GGUF model."
            ) from e

        if not os.path.exists(model_path):
            raise RuntimeError(
                f"Local model not found at '{model_path}'. "
                "Download a GGUF model and set LOCAL_MODEL_PATH in your .env."
            )

        llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )

        prompt = f"""<|system|>
{system}
<|user|>
{user}
<|assistant|>
"""
        out = llm(prompt, max_tokens=1200, temperature=self.temperature, stop=["<|user|>", "<|system|>"])
        return out["choices"][0]["text"].strip()


async def llm_json_list(client: LLMClient, system: str, user: str) -> List[str]:
    """Ask the model to output strict JSON: {"questions": [...]} and parse safely."""
    raw = await client.complete(system, user)
    obj: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json.loads(raw[start : end + 1])
    if not obj or "questions" not in obj or not isinstance(obj["questions"], list):
        raise RuntimeError("Model did not return valid JSON with a 'questions' list.")
    return [str(q).strip() for q in obj["questions"] if str(q).strip()]
