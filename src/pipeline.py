from utils import load_function_definitions, load_prompts
from llm_sdk import Small_LLM_Model
from decoder import select_function_name, build_number_params
from models import FunctionCallResult
import json
import re
def build_params(
            model: Small_LLM_Model,
            prompt: str,
            function
        ):
    extracted = {}
    digits = re.findall(r"-?\d+\.?\d*", prompt)
    digits = [int(d) for d in digits]
    # Group params by type
    number_params = {
        name: param
        for name, param in function.parameters.items()
        if param.type == "number"
    }
    string_params = {
        name: param
        for name, param in function.parameters.items()
        if param.type == "string"
    }

    # Each extractor gets the full function so the LLM sees the complete schema
    if number_params:
        number_results = build_number_params(model, prompt, function, digits)
        extracted.update(number_results)

    # if string_params:
    #     string_results = build_string_params(model, prompt, function)
    #     extracted.update(string_results)
    # print(number_params)
    # print(string_params)
    # print("extracted params:", extracted)
    return extracted
    # params = build_number_params(model, prompt, function, digits)
    # print(digits)
    # print("params :", params) 
    # print(function.parameters)

def run_pipeline(function_path: str, input_path: str, output_path: str) -> int:
    functions = load_function_definitions(function_path)
    prompts = load_prompts(input_path)

    if not functions or not prompts:
        with open(output_path, 'w') as f:
            json.dump([], f)
        return 1
    model = Small_LLM_Model()
    # id_to_text = build_id_to_text(model)
    # tokens_sets = build_token_sets(id_to_text)
    results: list[FunctionCallResult] = []
    # print("TEST 1", model.encode("hello world"))
    # print("TEST 2", model.encode("hello world").tolist())
    for prompt in prompts:
        print("prompt:", prompt.prompt)
        name = select_function_name(model, prompt.prompt, functions)
        print("selected function:", name)
        function = [f for f in functions if f.name == name][0]
        # print("function", function)
        # params = build_params(prompt.prompt, function)
        params = build_params(model, prompt.prompt, function)
        result = {"prompt": prompt.prompt,"name": name, "parameters": params}
        # print(result)
        results.append(result)
        # print("function name: ", name)

    with open(output_path, 'w') as f:
        # Use json.dumps() to convert a Python object (like dict) into a JSON string.
        # Use json.loads() to convert a JSON string into a Python object.
        # indent format json
        json.dump([res for res in results], f, indent=4)
    return 0
