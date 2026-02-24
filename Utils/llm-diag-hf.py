import os
from openai import OpenAI

# HF_TOKEN should be your Hugging Face access token
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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",
)

completion = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct", #"Qwen/Qwen2.5-7B-Instruct-1M",  # conversational LLM from HF's recommended list
    messages=[
        {"role": "user", "content": "Say hello in a short sentence."}
    ],
    max_tokens=64,
)

print(completion.choices[0].message)

