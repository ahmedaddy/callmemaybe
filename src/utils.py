import os
import json
import sys
from config import (
    DEFAULT_FUNCTIONS_FILE,
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_FILE,
    FALLBACK_FUNCTIONS_FILE,
)
from pydantic import ValidationError
from typing import Any
from models import FunctionDefinition, PromptItem


def read_json_file(path: str) -> Any | None:
    try:
        with open(path, "r", encoding="UTF-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
    except Exception:
        print("There is and error in file reading")
    return None


def load_function_definitions(path: str):
    data = read_json_file(path)
    if data is None:
        return []
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as e:
        print(
                f"Error: invalid function definitions in {path}: {e}",
                file=sys.stderr)
        return []


def load_prompts(path: str) -> list[PromptItem]:
    data = read_json_file(path)
    # if data in None:
    #     return []
    try:
        return [PromptItem.model_validate(item) for item in data]
    except ValidationError as e:
        print(f"Error: invalid prompt list in {path}: {e}", file=sys.stderr)
        return []


def resolve_functions_path(root: str, custom: str | None) -> str:
    if custom is not None:
        return custom
    primary = os.path.join(root, "data", "input", DEFAULT_FUNCTIONS_FILE)
    if os.path.exists(primary):
        return primary
    return os.path.join(root, "data", "input", FALLBACK_FUNCTIONS_FILE)


def resolve_input_path(root: str, custum: str | None):
    if custum is not None:
        return custum
    return os.path.join(root, "data", "input", DEFAULT_INPUT_FILE)


def resolve_output_path(root: str, custum: str | None):
    if custum is not None:
        return custum
    return os.path.join(root, "data", "output", DEFAULT_OUTPUT_FILE)
