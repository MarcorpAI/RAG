from dataclasses import dataclass

import httpx


class GenerationConfigError(RuntimeError):
    pass


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetrievedContext:
    chunk_index: int
    text: str
    score: float


def build_prompt(question: str, contexts: list[RetrievedContext]) -> str:
    rendered_context = "\n\n".join(
        f"[chunk {item.chunk_index} | score {item.score:.4f}]\n{item.text}"
        for item in contexts
    )
    return (
        "You are a precise document assistant. Answer the user's question using ONLY "
        "the context provided below. If the answer is not in the context, say "
        '"I don\'t know based on the provided document."\n\n'
        "--- CONTEXT ---\n"
        f"{rendered_context}\n"
        "--- END CONTEXT ---\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


class HuggingFaceGenerator:
    def __init__(self, api_key: str | None, model: str, timeout_seconds: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def generate(self, question: str, contexts: list[RetrievedContext]) -> str:
        if not self.api_key:
            raise GenerationConfigError("HF_API_KEY must be set to generate answers")

        prompt = build_prompt(question, contexts)
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 256,
            "temperature": 0.1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            raise GenerationError("Hugging Face generation request failed") from exc

        if response.status_code >= 400:
            raise GenerationError(
                f"Hugging Face generation failed with status {response.status_code}"
            )

        data = response.json()
        return _extract_generated_text(data).strip()


def _extract_generated_text(data) -> str:
    if isinstance(data, dict):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = message.get("content")
                if content:
                    return str(content)
        if "generated_text" in data:
            return str(data["generated_text"])
        if "error" in data:
            raise GenerationError(str(data["error"]))
    raise GenerationError("Unexpected Hugging Face generation response")
