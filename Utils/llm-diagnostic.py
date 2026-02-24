import os
import requests
from dotenv import load_dotenv

# -------------------------------------------------
# Load .env (system env vars override)
# -------------------------------------------------
load_dotenv()


def result(ok: bool, reason: str):
    return {"ok": ok, "reason": reason}


# -------------------------------------------------
# OpenAI Diagnostic (real conversational test)
# -------------------------------------------------
def check_openai() -> dict:
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return result(False, "Missing OPENAI_API_KEY")

    try:
        client = OpenAI(api_key=key)

        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return result(True, "OK")

    except Exception as e:
        msg = str(e).lower()

        if "rate limit" in msg or "429" in msg:
            return result(False, "Rate limited")
        if "insufficient" in msg:
            return result(False, "Insufficient credits")
        if "invalid" in msg or "401" in msg:
            return result(False, "Invalid API key")

        return result(False, f"Connection or other error: {e}")


# -------------------------------------------------
# Anthropic Diagnostic
# -------------------------------------------------
def check_anthropic() -> dict:
    from anthropic import Anthropic

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return result(False, "Missing ANTHROPIC_API_KEY")

    try:
        client = Anthropic(api_key=key)
        client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return result(True, "OK")

    except Exception as e:
        msg = str(e).lower()

        if "credit balance" in msg or "insufficient" in msg:
            return result(False, "Insufficient credits")
        if "invalid" in msg or "401" in msg:
            return result(False, "Invalid API key")

        return result(False, f"Connection or other error: {e}")


# -------------------------------------------------
# Google Gemini Diagnostic (new google.genai SDK)
# -------------------------------------------------
def check_google() -> dict:
    import google.genai as genai

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return result(False, "Missing GOOGLE_API_KEY")

    try:
        client = genai.Client(api_key=key)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="ping"
        )

        return result(True, "OK")

    except Exception as e:
        msg = str(e).lower()

        if "quota" in msg or "insufficient" in msg:
            return result(False, "Insufficient credits")
        if "invalid" in msg or "permission" in msg:
            return result(False, "Invalid API key")

        return result(False, f"Connection or other error: {e}")


# -------------------------------------------------
# Grok (xAI) Diagnostic — conversational validation
# -------------------------------------------------
def check_grok() -> dict:
    import openai  # xAI uses OpenAI-compatible SDK

    key = os.getenv("GROK_API_KEY")
    if not key:
        return result(False, "Missing GROK_API_KEY")

    try:
        client = openai.OpenAI(
            api_key=key,
            base_url="https://api.x.ai/v1",
        )

        # Conversational test matching your working curl call
        client.chat.completions.create(
            model="grok-4-latest",
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."}
            ],
            max_tokens=20,
            temperature=0,
            stream=False,
        )

        return result(True, "OK")

    except Exception as e:
        msg = str(e).lower()

        if "invalid" in msg or "401" in msg:
            return result(False, "Invalid API key")
        if "quota" in msg or "insufficient" in msg:
            return result(False, "Insufficient credits")
        if "rate" in msg or "429" in msg:
            return result(False, "Rate limited")

        return result(False, f"Connection or other error: {e}")



# -------------------------------------------------
# Hugging Face Router Chat Diagnostic (correct setup)
# -------------------------------------------------
def check_huggingface() -> dict:
    from openai import OpenAI

    key = os.getenv("HF_TOKEN")
    if not key:
        return result(False, "Missing HF_TOKEN")

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=key,
        )

        client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=20,
        )

        return result(True, "OK")

    except Exception as e:
        msg = str(e).lower()

        if "model_not_supported" in msg:
            return result(False, "Model not supported by HF Router")
        if "invalid" in msg or "401" in msg:
            return result(False, "Invalid API key")
        if "quota" in msg or "insufficient" in msg:
            return result(False, "Insufficient credits")

        return result(False, f"Connection or other error: {e}")


# -------------------------------------------------
# Run All Diagnostics
# -------------------------------------------------
def run_all():
    checks = {
        "openai": check_openai(),
        "anthropic": check_anthropic(),
        "google": check_google(),
        "grok": check_grok(),
        "huggingface": check_huggingface(),
    }

    print("\n=== LLM PROVIDER DIAGNOSTICS ===\n")
    for provider, outcome in checks.items():
        status = "OK" if outcome["ok"] else "FAIL"
        print(f"{provider:<12} {status:<6} {outcome['reason']}")
    print()


if __name__ == "__main__":
    run_all()
