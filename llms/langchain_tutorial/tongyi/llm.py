from __future__ import annotations

import asyncio
import functools
import logging
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from requests.exceptions import HTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

from langchain.llms import Tongyi


class CustomTongyi(Tongyi):
    """Add Features:
        1. Support tracing tokens usage when calling stream()"""

    def _generation_from_qwen_resp(
            self, resp: Any, is_last_chunk: bool = True
    ) -> Dict[str, Any]:
        # According to the response from dashscope,
        # each chunk's `generation_info` overwrites the previous one.
        # Besides, The `merge_dicts` method,
        # which is used to concatenate `generation_info` in `GenerationChunk`,
        # does not support merging of int type values.
        # Therefore, we adopt the `generation_info` of the last chunk
        # and discard the `generation_info` of the intermediate chunks.
        if is_last_chunk:
            return dict(
                text=resp["output"]["text"],
                generation_info=dict(
                    finish_reason=resp["output"]["finish_reason"],
                    request_id=resp["request_id"],
                    token_usage=dict(resp["usage"]),
                    model_name=self.model_name
                ),
            )
        else:
            return dict(text=resp["output"]["text"])
