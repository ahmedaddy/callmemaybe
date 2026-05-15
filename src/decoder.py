from llm_sdk import Small_LLM_Model
from utils import read_json_file
from models import TokenSets, FunctionDefinition


def build_function_selection_prompt(prompt: str,
                                    functions: list[FunctionDefinition]):
    lines = [
        "You are a function selection system.",
        "",
        "Your job is to select ONE function that best solves the user's request.",
        "",
        "IMPORTANT RULES:",
        "- Return ONLY the function name.",
        "- Return EXACTLY ONE function.",
        "- Do NOT explain your choice.",
        "- Do NOT add extra words or symbols.",
        "- Do NOT modify function names.",
        "",
        "HOW TO CHOOSE:",
        "- Read the FULL user request carefully.",
        "- Understand the user's real intent.",
        "- Do NOT rely only on keywords.",
        "- Some words may have different meanings depending on context.",
        "- Choose the function that solves the COMPLETE task.",
        "- Avoid functions that only partially match the request.",
        "",
        f"USER REQUEST:\n{prompt}",
        "",
        "AVAILABLE FUNCTIONS:"
    ]

    for f in functions:
        lines.append(f"- {f.name}: {f.description}")

    lines.append("")
    lines.append("Return ONLY one function name.")
    return "\n".join(lines)

def encode_ids(model: Small_LLM_Model, selecting_prompt: str):
    ids = model.encode(selecting_prompt).tolist()
    if not ids:
        return []
    return [int(x) for x in ids[0]]

def pick_best_token(logits: list[float], allowed_next) -> int | None:
    """Pick the token id with the highest logit among allowed_next."""

    best_token: int | None = None
    best_logit: float | None = None
    for token_id in allowed_next:
        if token_id >= len(logits):
            continue
        logit = logits[token_id]
        if best_logit is None or logit > best_logit:
            best_logit = logit
            best_token = token_id
    return best_token

def constrained_choice(
    model: Small_LLM_Model,
    prompt_ids: list[int],
    choices: list[str],
) -> str:
    """Choose one of *choices* using greedy constrained decoding."""

    generated: list[int] = []
    candidates_map = {choice: encode_ids(model, choice) for choice in choices}
    max_len = max((len(ids) for ids in candidates_map.values()), default=0)
    for _ in range(max_len):
        remaining = [
            (name, ids)
            for name, ids in candidates_map.items()
            if ids[: len(generated)] == generated
        ]
        if not remaining:
            break
        exact = [name for name, ids in remaining if len(ids) == len(generated)]
        if len(remaining) == 1 and exact:
            return exact[0]
        allowed_next = {
            ids[len(generated)]
            for _, ids in remaining
            if len(ids) > len(generated)
        }
        if not allowed_next:
            break
        logits = model.get_logits_from_input_ids(prompt_ids + generated)
        next_token = pick_best_token(logits, allowed_next)
        if next_token is None:
            break
        generated.append(next_token)
    
    for name, ids in candidates_map.items():
        if ids == generated:
            return name
    return choices[0] if choices else ""

def select_function_name(model: Small_LLM_Model, prompt: str, functions: list[FunctionDefinition]):
    if not functions:
        return ""
    selecting_prompt = build_function_selection_prompt(prompt, functions)
    prompt_ids = encode_ids(model, selecting_prompt)
    valid_functions = [f.name for f in functions]
    selected_function = constrained_choice(
        model=model,
        prompt_ids=prompt_ids,
        choices=valid_functions,
    )
    return selected_function

def build_number_params(model: Small_LLM_Model, question, function, digits):
    # Only number params
    number_param_names = [
        name for name, param in function.parameters.items()
        if param.type == "number"
    ]

    if not number_param_names:
        return {}

    prompt = f"""
    You are a number extraction system.

    IMPORTANT:
    - DO NOT solve the question
    - DO NOT do any math
    - DO NOT interpret meaning

    Only extract numbers exactly as they appear in the text.
    Extract exactly {len(number_param_names)} number(s), in the order they appear.

    Question:
    {question}

    Output format (comma separated, nothing else):
    number1,number2,...
    """

    prompt_ids = encode_ids(model, prompt)
    generated = []

    allowed_ids = set()
    comma_id = encode_ids(model, ",")[0]
    allowed_ids.add(comma_id)

    for d in digits:
        for token_id in encode_ids(model, str(d)):
            allowed_ids.add(token_id)

    max_tokens = sum(len(encode_ids(model, str(d))) for d in digits) + len(digits)

    for _ in range(max_tokens + 2):
        logits = model.get_logits_from_input_ids(prompt_ids + generated)
        next_token = pick_best_token(logits, allowed_ids)

        if next_token is None:
            break

        decoded = model.decode([next_token])
        if decoded in ("\n", "<END>", ""):
            break

        generated.append(next_token)

    gen = model.decode(generated)
    print("Generated text:", gen)

    numbers = [num.strip() for num in gen.split(",") if num.strip()]
    print("Extracted numbers:", numbers)
    result = {}
    for name in number_param_names:
        if numbers:
            result[name] = numbers.pop(0)

    return result