from dataclasses import dataclass

from app.config import Settings
from app.pipeline.embedder import SentenceTransformerEmbedder
from app.pipeline.generator import HuggingFaceGenerator
from app.pipeline.vector_store import FaissDocumentStore


@dataclass
class AppState:
    settings: Settings
    embedder: SentenceTransformerEmbedder
    generator: HuggingFaceGenerator
    vector_store: FaissDocumentStore

    @classmethod
    def from_settings(cls, settings: Settings) -> "AppState":
        data_dir = settings.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
        return cls(
            settings=settings,
            embedder=embedder,
            generator=HuggingFaceGenerator(
                api_key=settings.hf_api_key,
                model=settings.generation_model,
                timeout_seconds=settings.hf_timeout_seconds,
            ),
            vector_store=FaissDocumentStore(data_dir=data_dir),
        )
