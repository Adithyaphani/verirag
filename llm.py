from huggingface_hub import InferenceClient
import config

_client = InferenceClient(model=config.LLM_MODEL, token=config.HF_TOKEN)


def chat(system, user, max_tokens=600, temperature=0.2):
    r = _client.chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content.strip()
