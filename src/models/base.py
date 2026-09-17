"""Model interface. Every backend (built-in transformer, Hugging Face, Nemotron, API) implements this contract,
so the data pipeline never depends on a particular model. Implementations arrive in Phases 5 and 7."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ModelAdapter(ABC):
    key: str = "base"
    label: str = "Base model"

    def __init__(self, config: dict, task: str):
        self.config, self.task = config, task

    @classmethod
    def availability(cls, hardware: dict, config: dict) -> tuple[bool, str]:
        """Whether this backend can run on the detected hardware, with a message for the UI."""
        return False, "not implemented yet"

    @abstractmethod
    def train(self, prepared, settings: dict, progress=None) -> dict: ...

    @abstractmethod
    def predict(self, frame): ...

    @abstractmethod
    def evaluate(self, prepared) -> dict: ...

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

    @abstractmethod
    def load(self, path: str | Path) -> "ModelAdapter": ...


# Backends planned for the UI; status is shown honestly until each is implemented.
PLANNED_BACKENDS = [
    {"key": "sanity_transformer", "label": "Built-in Sanity Transformer", "phase": 5,
     "description": "Small tabular transformer trained from scratch; runs on CPU."},
    {"key": "huggingface", "label": "Hugging Face Model", "phase": 7,
     "description": "Configurable model from the Hugging Face Hub through an adapter."},
    {"key": "nemotron", "label": "Nemotron", "phase": 7,
     "description": "Small NVIDIA Nemotron checkpoint (configurable); requires a CUDA GPU."},
    {"key": "api", "label": "Custom/API Model", "phase": 7,
     "description": "Your own model behind an HTTP endpoint."},
]
