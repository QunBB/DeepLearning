from typing import (
    Any,
    Callable,
    Dict,
    Type,
    Union
)

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool


def convert_to_openai_tool(function: Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool],
) -> Dict[str, Any]:
    function = convert_to_openai_function(function)

    tool = {"type": "function",
            "function": {
                "name": function["name"],
                "description": function["description"],
                "parameters": function["parameters"]
            }}

    if "required" in function:
        tool["required"] = function["function"]["required"]

    return tool
