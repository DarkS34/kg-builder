from collections.abc import Callable

from json_repair import repair_json
from loguru import logger

from .prompts import json_repair_prompt


def parse_json_object(text: str) -> tuple[dict | None, str | None]:
    raw = repair_json(text, return_objects=True)
    if not isinstance(raw, dict):
        return None, "model did not return a JSON object"
    return raw, None


def parse_with_repair(
    response: str,
    parse: Callable[[str], tuple[object | None, str | None]],
    engine,
    repair_model: str,
    max_attempts: int,
    shape: str = "object",
    log_prefix: str = "",
) -> tuple[object | None, str | None]:
    result, error = parse(response)

    for attempt in range(1, max_attempts + 1):
        if result is not None:
            return result, None

        logger.warning(
            f"{log_prefix}repair {attempt}/{max_attempts}: {str(error).replace(chr(10), ' | ')}"
        )
        prompt = json_repair_prompt(
            broken_output=response, error_msg=error or "invalid JSON", shape=shape
        )
        response = engine.generate(model=repair_model, prompt=prompt).response
        result, error = parse(response)

    return result, error
