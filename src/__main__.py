import argparse
import os
from utils import (
                    resolve_functions_path,
                    resolve_input_path,
                    resolve_output_path,
                    read_json_file)
from pipeline import run_pipeline
from llm_sdk import Small_LLM_Model
from decoder import select_function_name

# Default paths
DEFAULT_FUNCTIONS_FILE = "data/input/functions_definition.json"
DEFAULT_INPUT_FILE = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT_FILE = "data/output/function_calling_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Function calling system using constrained decoding",
        epilog="""
Examples:
  uv run python -m src
  uv run python -m src --input custom_prompts.json
  uv run python -m src --functions_definition funcs.json --output results.json
        """,
    )
    parser.add_argument(
        "--functions_definition", type=str, default=DEFAULT_FUNCTIONS_FILE
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE
    )

    return parser.parse_args()

def encode_ids(model: Small_LLM_Model, text: str) -> list[int]:
    """Encode *text* into a flat list of token ids."""

    ids = model.encode(text).tolist()
    if not ids:
        return []
    return [int(x) for x in ids[0]]

def main() -> None:
    args = parse_args()
    root = os.getcwd()
    # print(args.functions_definition)
    function_path = resolve_functions_path(root, args.functions_definition)
    input_path = resolve_input_path(root, args.input)
    output_path = resolve_output_path(root, args.output)

    functions = read_json_file(function_path)
    input_file = read_json_file(input_path)
    model = Small_LLM_Model()
    # prompt = "What is the sum of 2 and 3?"
    # print("question:", question)
    # print("function str = ",function_file)
    # selected_function = select_function_name(model, prompt, functions)
    # print("Selected Function: ",selected_function)
    # print(model.encode("hello world"))
    # print(model.encode(" hello world"))
    # print(model.encode("- hello world"))
    return run_pipeline(function_path, input_path, output_path)
    # data = load_function_definitions(args.functions_definition)
    # print(data)


if __name__ == "__main__":
    main()

