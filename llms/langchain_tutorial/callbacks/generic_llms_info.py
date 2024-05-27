"""Callback Handler that prints to std out."""
import threading
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult, Generation

# CYN Unit
MODEL_COST_PER_1K_TOKENS = {
    # Tongyi input
    "qwen-plus": 0.02,
    "qwen-turbo": 0.008,
    # Tongyi output
    "qwen-plus-completion": 0.02,
    "qwen-turbo-completion": 0.02,
}


def standardize_model_name(
    model_name: str,
    is_completion: bool = False,
) -> str:
    """
    Standardize the model name to a format that can be used in the generic LLMs API.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Standardized model name.

    """
    model_name = model_name.lower()
    if is_completion:
        return model_name + "-completion"
    else:
        return model_name


def get_llm_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    Get the cost in CYN for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in CYN.
    """
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid LLM model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


class GenericLLMsCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks generic LLMs info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (CYN): ¥{self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""

        model_name = ""
        token_usage = {}
        if response.llm_output is not None:
            token_usage = response.llm_output.get("token_usage", {})
            model_name = response.llm_output.get("model_name", "")

        if not token_usage:
            if response.generations and response.generations[-1] and "token_usage" in response.generations[-1][-1].generation_info:
                token_usage = response.generations[-1][-1].generation_info["token_usage"]
                if not model_name:
                    model_name = response.generations[-1][-1].generation_info.get("model_name", "")  # You Need to return `model_name` in generation_info when calling stream()

        if not token_usage:
            with self._lock:
                self.successful_requests += 1

            return None
        
        model_name = standardize_model_name(model_name)

        # compute tokens and cost for this request
        completion_tokens = token_usage.get("completion_tokens") or token_usage.get("output_tokens", 0)  # Some LLMs e.g. Anthropic,Tongyi use `output_tokens` rather than `completion_tokens`
        prompt_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens", 0)  # Some LLMs e.g. Anthropic,Tongyi use `input_tokens` rather than `prompt_tokens`
        if model_name in MODEL_COST_PER_1K_TOKENS:
            completion_cost = get_llm_token_cost_for_model(
                model_name, completion_tokens, is_completion=True
            )
            prompt_cost = get_llm_token_cost_for_model(model_name, prompt_tokens)
        else:
            completion_cost = 0
            prompt_cost = 0

        # update shared state behind lock
        with self._lock:
            self.total_cost += prompt_cost + completion_cost
            self.total_tokens += token_usage.get("total_tokens", 0)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "GenericLLMsCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "GenericLLMsCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
