import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    text: str
    start_token: int
    end_token: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = tokenize(text)
    if not tokens:
        return []

    chunks: list[TextChunk] = []
    step = chunk_size - overlap
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(
            TextChunk(
                chunk_index=len(chunks),
                text=" ".join(tokens[start:end]),
                start_token=start,
                end_token=end,
            )
        )
        if end == len(tokens):
            break
        start += step

    return chunks
