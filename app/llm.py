from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Literal

import httpx


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    mode: Literal["local", "openai_compat", "anthropic", "google_genai", "google_generativeai", "hf_inference"]
    api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None
    model_env: Optional[str] = None
    default_model: Optional[str] = None


# --------------------------------------------------------------------
# Provider Registry
# --------------------------------------------------------------------
# Built-ins + a "generic provider" mechanism:
#   PROVIDER_<NAME>_API_KEY
#   PROVIDER_<NAME>_BASE_URL
#   PROVIDER_<NAME>_MODEL
#   PROVIDER_<NAME>_MODE=openai_compat|anthropic|google_genai|google_generativeai|hf_inference
#
# Example:
#   LLM_PROVIDER=deepseek
#   PROVIDER_DEEPSEEK_MODE=openai_compat
#   PROVIDER_DEEPSEEK_API_KEY=...
#   PROVIDER_DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
#   PROVIDER_DEEPSEEK_MODEL=deepseek-chat
# --------------------------------------------------------------------


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def resolve_provider() -> str:
    """Provider selection is explicit, but accept legacy env var names.

    Supported env vars:
      - LLM_PROVIDER (canonical)
      - LLM_Provider (legacy)
      - LLM_PROVIDER_NAME (alt)
    """
    provider = (_env("LLM_PROVIDER") or _env("LLM_Provider") or _env("LLM_PROVIDER_NAME") or "local").lower()
    if provider in {"auto", ""}:
        return "local"
    # normalize a few common aliases
    if provider in {"hf", "huggingface", "hugging_face"}:
        return "huggingface"
    if provider in {"xai", "grok"}:
        return "grok"
    return provider


def _builtin_provider_specs() -> Dict[str, ProviderSpec]:
    return {
        "local": ProviderSpec(
            name="local",
            mode="local",
            model_env="LOCAL_MODEL_NAME",
            default_model="local-gguf",
        ),
        "openai": ProviderSpec(
            name="openai",
            mode="openai_compat",
            api_key_env="OPENAI_API_KEY",
            base_url_env="OPENAI_BASE_URL",
            model_env="OPENAI_MODEL",
            default_model="gpt-4o-mini",
        ),
        "grok": ProviderSpec(
            name="grok",
            mode="openai_compat",
            api_key_env="GROK_API_KEY",
            base_url_env="GROK_BASE_URL",
            model_env="GROK_MODEL",
            default_model="grok-4-latest",
        ),
        # HF Router is OpenAI-compatible (matches your diagnostic script)
        "huggingface": ProviderSpec(
            name="huggingface",
            mode="openai_compat",
            api_key_env="HF_TOKEN",
            base_url_env="HF_BASE_URL",
            model_env="HF_MODEL",
            default_model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        "anthropic": ProviderSpec(
            name="anthropic",
            mode="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            model_env="ANTHROPIC_MODEL",
            default_model="claude-3-5-sonnet-latest",
        ),
        # Google (new) sdk: google.genai
        "google": ProviderSpec(
            name="google",
            mode="google_genai",
            api_key_env="GOOGLE_API_KEY",
            model_env="GOOGLE_MODEL",
            default_model="gemini-2.0-flash",
        ),
        # Backwards compatibility for older google.generativeai
        "google_generativeai": ProviderSpec(
            name="google_generativeai",
            mode="google_generativeai",
            api_key_env="GOOGLE_API_KEY",
            model_env="GOOGLE_MODEL",
            default_model="gemini-1.5-flash",
        ),
        # Keep the old HF Inference API as an explicit option
        "hf_inference": ProviderSpec(
            name="hf_inference",
            mode="hf_inference",
            api_key_env="HF_TOKEN",
            model_env="HF_MODEL",
            default_model="meta-llama/Llama-3.1-8B-Instruct",
        ),
    }


def _load_generic_provider_spec(provider: str) -> Optional[ProviderSpec]:
    """Allow any provider name using PROVIDER_<NAME>_* env vars."""
    key = provider.upper().replace("-", "_").replace(".", "_")
    mode = (_env(f"PROVIDER_{key}_MODE") or "openai_compat").lower()
    api_key_env = f"PROVIDER_{key}_API_KEY"
    base_url_env = f"PROVIDER_{key}_BASE_URL"
    model_env = f"PROVIDER_{key}_MODEL"
    # Only consider it a configured generic provider if at least the API key exists
    if os.getenv(api_key_env) is None and os.getenv(model_env) is None and os.getenv(base_url_env) is None:
        return None
    if mode not in {"openai_compat", "anthropic", "google_genai", "google_generativeai", "hf_inference"}:
        mode = "openai_compat"
    return ProviderSpec(
        name=provider,
        mode=mode,  # type: ignore[arg-type]
        api_key_env=api_key_env,
        base_url_env=base_url_env,
        model_env=model_env,
        default_model=None,
    )


def get_provider_spec(provider: str) -> ProviderSpec:
    specs = _builtin_provider_specs()
    if provider in specs:
        return specs[provider]
    generic = _load_generic_provider_spec(provider)
    if generic:
        return generic
    # Unknown provider -> fail safe to local
    return specs["local"]


def get_model_for(spec: ProviderSpec) -> str:
    if spec.model_env:
        v = _env(spec.model_env)
        if v:
            return v
    return spec.default_model or "gpt-4o-mini"


class LLMClient:
    def __init__(self) -> None:
        self.provider = resolve_provider()
        self.spec = get_provider_spec(self.provider)
        self.model = get_model_for(self.spec)
        self.temperature = float(_env("LLM_TEMPERATURE", "0.2") or "0.2")

    async def complete(self, system: str, user: str) -> str:
        mode = self.spec.mode
        if mode == "local":
            return await self._local_complete(system, user)
        if mode == "openai_compat":
            api_key = _env(self.spec.api_key_env or "")
            base_url = _env(self.spec.base_url_env or "", "https://api.openai.com/v1") or "https://api.openai.com/v1"

            # Special-case defaults
            if self.provider == "grok":
                base_url = _env("GROK_BASE_URL", "https://api.x.ai/v1") or "https://api.x.ai/v1"
            if self.provider == "huggingface":
                base_url = _env("HF_BASE_URL", "https://router.huggingface.co/v1") or "https://router.huggingface.co/v1"

            return await self._openai_compatible_complete(system, user, api_key=api_key, base_url=base_url)

        if mode == "anthropic":
            return await self._anthropic_complete(system, user)

        if mode == "google_genai":
            return await self._google_genai_complete(system, user)

        if mode == "google_generativeai":
            return await self._google_generativeai_complete(system, user)

        if mode == "hf_inference":
            return await self._hf_inference_complete(system, user)

        return await self._local_complete(system, user)

    async def _openai_compatible_complete(self, system: str, user: str, api_key: Optional[str], base_url: str) -> str:
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.provider}'. "
                f"Set {self.spec.api_key_env or '..._API_KEY'} in .env (or set LLM_PROVIDER=local)."
            )

        url = base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}

        max_tokens = int(_env("OPENAI_MAX_TOKENS", "900") or "900")
        timeout_s = float(_env("OPENAI_TIMEOUT_SECONDS", "60") or "60")
        max_retries = int(_env("OPENAI_MAX_RETRIES", "6") or "6")
        base_backoff = float(_env("OPENAI_BACKOFF_SECONDS", "1.0") or "1.0")

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        async with httpx.AsyncClient(timeout=timeout_s) as client:
            for attempt in range(max_retries + 1):
                r = await client.post(url, headers=headers, json=payload)

                if r.status_code < 400:
                    data = r.json()
                    return data["choices"][0]["message"]["content"]

                # Retry on rate limits and transient errors
                if r.status_code in (429, 500, 502, 503, 504):
                    retry_after = r.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except Exception:
                            wait = base_backoff
                    else:
                        wait = min(20.0, base_backoff * (2 ** attempt))

                    if attempt >= max_retries:
                        raise httpx.HTTPStatusError(
                            f"Provider '{self.provider}' error after retries: {r.status_code} {r.text}",
                            request=r.request,
                            response=r,
                        )
                    await asyncio.sleep(wait)
                    continue

                # ... inside the retry block ...
                if r.status_code == 429:
                    # 1) Prefer Retry-After header if present
                    retry_after = r.headers.get("retry-after")
                    wait = None
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except Exception:
                            wait = None

                    # 2) HF Router sometimes encodes wait time in body text
                    if wait is None:
                        m = re.search(r"wait\s+([0-9]*\.?[0-9]+)\s*s", r.text, re.IGNORECASE)
                        if m:
                            wait = float(m.group(1))

                    # 3) Fallback exponential backoff
                    if wait is None:
                        wait = min(20.0, base_backoff * (2 ** attempt))
                else:
                    wait = min(20.0, base_backoff * (2 ** attempt))


                # Non-retryable
                r.raise_for_status()

        raise RuntimeError("Unexpected: OpenAI-compatible request loop exited without returning.")

    async def _anthropic_complete(self, system: str, user: str) -> str:
        api_key = _env("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY (or set LLM_PROVIDER=local).")
        try:
            from anthropic import AsyncAnthropic  # type: ignore

            client = AsyncAnthropic(api_key=api_key)
            msg = await client.messages.create(
                model=self.model,
                max_tokens=int(_env("ANTHROPIC_MAX_TOKENS", "1200") or "1200"),
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
                "max_tokens": int(_env("ANTHROPIC_MAX_TOKENS", "1200") or "1200"),
                "temperature": self.temperature,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return "".join([blk.get("text", "") for blk in data.get("content", [])])

    async def _google_genai_complete(self, system: str, user: str) -> str:
        api_key = _env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or set LLM_PROVIDER=local).")
        import google.genai as genai  # type: ignore

        client = genai.Client(api_key=api_key)
        # google.genai doesn't have system_instruction the same way across all versions; fold into contents.
        contents = f"{system}\n\n{user}" if system else user
        response = client.models.generate_content(model=self.model, contents=contents)
        return getattr(response, "text", "") or str(response)

    async def _google_generativeai_complete(self, system: str, user: str) -> str:
        api_key = _env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY (or set LLM_PROVIDER=local).")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model, system_instruction=system)
        resp = model.generate_content(user, generation_config={"temperature": self.temperature})
        return getattr(resp, "text", "") or ""

    async def _hf_inference_complete(self, system: str, user: str) -> str:
        token = _env("HF_TOKEN")
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
        model_path = _env("LOCAL_MODEL_PATH", "./models/model.gguf") or "./models/model.gguf"
        n_ctx = int(_env("LOCAL_MODEL_N_CTX", "4096") or "4096")
        n_threads = int(_env("LOCAL_MODEL_N_THREADS", "8") or "8")

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
