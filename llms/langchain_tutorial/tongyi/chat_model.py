from __future__ import annotations

import json
import logging
from requests.exceptions import HTTPError
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Union,
    Sequence,
    Type
)

from langchain_community.chat_models.tongyi import (
    ChatTongyi,
    _create_retry_decorator
)
from langchain_community.llms.tongyi import (
    generate_with_last_element_mark,
)
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    ToolCall,
    InvalidToolCall
)
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import (
    ChatGenerationChunk,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from tools.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def convert_dict_to_message(
    _dict: Mapping[str, Any], is_chunk: bool = False
) -> Union[BaseMessage, BaseMessageChunk]:
    """Convert a dict to a message."""
    role = _dict["role"]
    content = _dict["content"]

    if role == "user":
        return (
            HumanMessageChunk(content=content)
            if is_chunk
            else HumanMessage(content=content)
        )
    elif role == "assistant":
        tool_calls = []
        invalid_tool_calls = []
        if "tool_calls" in _dict:
            additional_kwargs = {"tool_calls": _dict["tool_calls"]}

            for index, value in enumerate(_dict["tool_calls"]):
                if is_chunk:
                    try:
                        tool_calls.append(
                            {
                                "name": value["function"].get("name"),
                                "args": value["function"].get("arguments"),
                                "id": value.get("id"),
                                # Tongyi does not respond with index,
                                # use index in the list instead
                                "index": index,
                            }
                        )
                    except KeyError:
                        pass
                else:
                    try:
                        parsed_tool = parse_tool_call(value, return_id=True)
                        if parsed_tool:
                            tool_calls.append(parsed_tool)
                    except Exception as e:
                        invalid_tool_calls.append(make_invalid_tool_call(value, str(e)))
        else:
            additional_kwargs = {}

        return (
            AIMessageChunk(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_call_chunks=tool_calls,  # type: ignore[arg-type]
                id=_dict.get("id"),
            )
            if is_chunk
            else AIMessage(
                content=content,
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,  # type: ignore[arg-type]
                invalid_tool_calls=invalid_tool_calls,
            )
        )
    elif role == "system":
        return (
            SystemMessageChunk(content=content)
            if is_chunk
            else SystemMessage(content=content)
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return (
            ToolMessageChunk(
                content=_dict.get("content", ""),
                tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
                additional_kwargs=additional_kwargs,
            )
            if is_chunk
            else ToolMessage(
                content=_dict.get("content", ""),
                tool_call_id=_dict.get("tool_call_id"),  # type: ignore[arg-type]
                additional_kwargs=additional_kwargs,
            )
        )
    else:
        return (
            ChatMessageChunk(role=role, content=content)
            if is_chunk
            else ChatMessage(role=role, content=content)
        )


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dict."""

    message_dict: Dict[str, Any] = {}
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        # message_dict = {"role": "assistant", "content": message.content}
        message_dict["role"] = "assistant"
        message_dict["content"] = getattr(message, "content", "")
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        elif message.tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ]
        elif message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "tool_calls" in message_dict:
            message_dict["content"] = ""
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id
        message_dict["content"] = message.content
        message_dict["name"] = message.name or message.additional_kwargs.get("name")
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def check_response(resp: Any) -> Any:
    """Check the response from the completion call."""
    if resp["status_code"] == 200:
        return resp
    elif resp["status_code"] in [400, 401]:
        raise ValueError(
            f"status_code: {resp['status_code']} \n "
            f"code: {resp['code']} \n message: {resp['message']}"
        )
    else:
        raise HTTPError(
            f"HTTP error occurred: status_code: {resp['status_code']} \n "
            f"code: {resp['code']} \n message: {resp['message']}",
            response=resp,
        )


class CustomChatTongyi(ChatTongyi):

    def _chat_generation_from_qwen_resp(
        self, resp: Any, is_chunk: bool = False, is_last_chunk: bool = True
    ) -> Dict[str, Any]:
        # According to the response from dashscope,
        # each chunk's `generation_info` overwrites the previous one.
        # Besides, The `merge_dicts` method,
        # which is used to concatenate `generation_info` in `GenerationChunk`,
        # does not support merging of int type values.
        # Therefore, we adopt the `generation_info` of the last chunk
        # and discard the `generation_info` of the intermediate chunks.
        choice = resp["output"]["choices"][0]
        message = convert_dict_to_message(choice["message"], is_chunk=is_chunk)
        if is_last_chunk:
            return dict(
                message=message,
                generation_info=dict(
                    finish_reason=choice["finish_reason"],
                    request_id=resp["request_id"],
                    token_usage=dict(resp["usage"]),
                    model_name=self.model_name,
                ),
            )
        else:
            return dict(message=message)
    
    # 重写该方法，支持function calling
    def _invocation_params(
        self, messages: List[BaseMessage], stop: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        params = {**self._default_params, **kwargs}
        if stop is not None:
            params["stop"] = stop
        # According to the Tongyi official docs
        # https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope#d553cbbee6mxk,
        # `incremental_output` with `tools` is not supported yet
        if params.get("stream") and not params.get("tools"):
            params["incremental_output"] = True

        message_dicts = [convert_message_to_dict(m) for m in messages]

        # And the `system` message should be the first message if present
        system_message_indices = [
            i for i, m in enumerate(message_dicts) if m["role"] == "system"
        ]
        if len(system_message_indices) == 1 and system_message_indices[0] != 0:
            raise ValueError("System message can only be the first message.")
        elif len(system_message_indices) > 1:
            raise ValueError("There can be only one system message at most.")

        params["messages"] = message_dicts

        return params

    def subtract_client_response(self, resp: Any, prev_resp: Any) -> Any:
        """Subtract prev response from curr response.

        Useful when streaming without `incremental_output = True`
        """

        resp_copy = json.loads(json.dumps(resp))
        choice = resp_copy["output"]["choices"][0]
        message = choice["message"]

        prev_resp_copy = json.loads(json.dumps(prev_resp))
        prev_choice = prev_resp_copy["output"]["choices"][0]
        prev_message = prev_choice["message"]

        message["content"] = message["content"].replace(prev_message["content"], "")

        if message.get("tool_calls"):
            for index, tool_call in enumerate(message["tool_calls"]):
                function = tool_call["function"]

                if prev_message.get("tool_calls"):
                    prev_function = prev_message["tool_calls"][index]["function"]

                    function["name"] = function["name"].replace(
                        prev_function["name"], ""
                    )
                    function["arguments"] = function["arguments"].replace(
                        prev_function["arguments"], ""
                    )

        return resp_copy

    # Tongyi使用tools不支持流式，需要进行特殊处理
    # {"status_code": 200, "request_id": "fb9b9135-cdc7-9c7a-85ff-27ccb128870a", "code": "", "message": "", "output": {"text": null, "finish_reason": null, "choices": [{"finish_reason": "null", "message": {"role": "assistant", "content": "我是"}}]}, "usage": {"input_tokens": 253, "output_tokens": 1, "total_tokens": 254}}
    # {"status_code": 200, "request_id": "fb9b9135-cdc7-9c7a-85ff-27ccb128870a", "code": "", "message": "", "output": {"text": null, "finish_reason": null, "choices": [{"finish_reason": "null", "message": {"role": "assistant", "content": "我是通"}}]}, "usage": {"input_tokens": 253, "output_tokens": 2, "total_tokens": 255}}
    # {"status_code": 200, "request_id": "fb9b9135-cdc7-9c7a-85ff-27ccb128870a", "code": "", "message": "", "output": {"text": null, "finish_reason": null, "choices": [{"finish_reason": "null", "message": {"role": "assistant", "content": "我是通义"}}]}, "usage": {"input_tokens": 253, "output_tokens": 3, "total_tokens": 256}}
    # {"status_code": 200, "request_id": "fb9b9135-cdc7-9c7a-85ff-27ccb128870a", "code": "", "message": "", "output": {"text": null, "finish_reason": null, "choices": [{"finish_reason": "null", "message": {"role": "assistant", "content": "我是通义千"}}]}, "usage": {"input_tokens": 253, "output_tokens": 4, "total_tokens": 257}}
    def stream_completion_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self)

        @retry_decorator
        def _stream_completion_with_retry(**_kwargs: Any) -> Any:
            responses = self.client.call(**_kwargs)
            prev_resp = None

            for resp in responses:
                # If we are streaming without `incremental_output = True`,
                # we need to calculate the delta response manually
                if _kwargs.get("stream") and not _kwargs.get(
                        "incremental_output", False
                ):
                    if prev_resp is None:
                        delta_resp = resp
                    else:
                        delta_resp = self.subtract_client_response(resp, prev_resp)
                    prev_resp = resp
                    yield check_response(delta_resp)
                else:
                    yield check_response(resp)

        return _stream_completion_with_retry(**kwargs)

    def _stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params: Dict[str, Any] = self._invocation_params(
            messages=messages, stop=stop, stream=True, **kwargs
        )

        for stream_resp, is_last_chunk in generate_with_last_element_mark(
                self.stream_completion_with_retry(**params)
        ):
            choice = stream_resp["output"]["choices"][0]
            message = choice["message"]
            if (
                    choice["finish_reason"] == "null"
                    and message["content"] == ""
                    and "tool_calls" not in message
            ):
                continue

            chunk = ChatGenerationChunk(
                **self._chat_generation_from_qwen_resp(
                    stream_resp, is_chunk=True, is_last_chunk=is_last_chunk
                )
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)
