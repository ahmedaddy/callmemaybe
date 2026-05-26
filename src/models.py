from pydantic import BaseModel, field_validator
from typing import Dict, Any


class ParameterDefinition(BaseModel):
    type: str


class ReturnDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("parameters")
    def check_not_empty(cls, v):
        if not v:
            raise ValueError("Function must have at least one parameter")
        return v


class FunctionTest(BaseModel):
    prompt: str


class PromptItem(BaseModel):
    prompt: str


class TokenSets(BaseModel):
    string_tokens: list[int]
    number_tokens: list[int]


class FunctionCallResult(BaseModel):
    """Structured output for a single prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any]
