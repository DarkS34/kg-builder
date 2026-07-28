from dataclasses import dataclass

import httpx
import ollama
from loguru import logger
from tqdm import tqdm


class InferenceError(Exception):
    pass


@dataclass
class GenerationResponse:
    response: str
    thinking: str | None = None


class OllamaEngine:
    name = "ollama"

    def __init__(self, host: str, timeout: float = 600.0):
        self.host = host
        self._client = ollama.Client(host=host, timeout=timeout)

    def is_available(self) -> bool:
        try:
            return httpx.get(self.host, timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def generate(self, model: str, prompt: str, think: bool | None = None) -> GenerationResponse:
        options = {} if think is None else {"think": think}
        try:
            response = self._client.generate(model=model, prompt=prompt, **options)
        except (ollama.ResponseError, httpx.RequestError) as e:
            raise InferenceError(f"Ollama generation failed for model '{model}': {e}") from e
        return GenerationResponse(
            response=response.response, thinking=getattr(response, "thinking", None)
        )

    def ensure_model(self, model: str) -> bool:
        try:
            installed = [info["model"] for info in self._client.list()["models"]]
        except (ollama.ResponseError, httpx.RequestError) as e:
            raise InferenceError(f"Could not list installed models: {e}") from e
        return True if model in installed else self._pull(model)

    def warmup(self, model: str) -> None:
        self.generate(model, "")

    def _pull(self, model: str) -> bool:
        try:
            logger.info(f"Downloading model '{model}'...")
            progress_bar = None
            for progress in self._client.pull(model, stream=True):
                total = progress.get("total") or 0
                completed = progress.get("completed") or 0
                if total > 0:
                    if progress_bar is None:
                        progress_bar = tqdm(total=total, unit="B", unit_scale=True, desc=model)
                    progress_bar.update(completed - progress_bar.n)
            if progress_bar is not None:
                progress_bar.close()
            logger.success(f"Successfully downloaded model '{model}'")
            return True
        except (ollama.ResponseError, httpx.RequestError) as e:
            logger.error(f"Download of '{model}' failed: {e}")
            return False


ENGINES = {OllamaEngine.name: OllamaEngine}


def create_engine(config) -> OllamaEngine:
    if config.inference_engine not in ENGINES:
        raise InferenceError(
            f"Unknown inference engine '{config.inference_engine}'. "
            f"Available: {', '.join(sorted(ENGINES))}"
        )
    return ENGINES[config.inference_engine](config.ollama_host, timeout=config.request_timeout)
