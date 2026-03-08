from anthropic import Anthropic
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from . import LLM, LLMResponse
from ..agent.tooling import ToolCall
from ..agent.tooling import Tool, ToolCall

class AnthropicLLM(LLM):

    HAS_COST = True
    SYSTEM_ROLE_NAME = 'system'

    DEFAULT_MAX_TOKENS = 64_000
    DEFAULT_MAX_THINKING_TOKENS = 10_000

    def __init__(self, model_name: str, timeout=None, *args, **kwargs):
        self.model_name = model_name
        self.client = Anthropic(timeout=timeout, *args, **kwargs)

    def _prepare_messages(self, messages: List[Dict[str, Any]]):
        system_prompt = None
        prepared_messages = messages.copy()

        if prepared_messages and prepared_messages[0].get("role") == self.SYSTEM_ROLE_NAME:
            system_prompt = prepared_messages.pop(0).get("content")

        return system_prompt, prepared_messages

    def _parse_thinking(self, think: Any, max_thinking_tokens: int) -> Dict[str, Any]:

        if not think:
            return {}

        if type(think) == bool:
            return {"thinking": {"type": "enabled", "budget_tokens": max_thinking_tokens}}

        if think == 'adaptive':
            return {"thinking": {"type": "adaptive"}}
        else:
            return {'output_config': {'effort': think}}


    @staticmethod
    def _filter_messages(content):
        thinking, text, structured, tool_calls = [], [], [], []
        for message in content:
            mtype = message['type']
            if mtype == 'thinking':
                thinking.append(message[mtype])
                continue
            if mtype == 'tool_use':
                tool_calls.append(message)
                continue
            if mtype == 'text':
                if 'parsed_output' in message:
                    structured.append(message['parsed_output'])
                else:
                    text.append(message[mtype])
        return thinking, text, structured, tool_calls

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float=1.,
        max_tokens: Optional[int] = DEFAULT_MAX_TOKENS,
        max_thinking_tokens= DEFAULT_MAX_THINKING_TOKENS,
        format: Optional[BaseModel] = None,
        think: Any = True,
        tools: Optional[List[Dict[str, Any]]] = [],
        enable_prompt_cache: bool = False,
        **kwargs
        ) -> LLMResponse:

        if enable_prompt_cache:
            kwargs['cache_control'] = {"type": "ephemeral"}

        think = self._parse_thinking(think, max_thinking_tokens)

        system_prompt, other_messages = self._prepare_messages(messages) 

        if format:
            gen_function = self.client.messages.parse
            kwargs['output_format'] = format
        else:
            gen_function = self.client.messages.create

        raw_response = gen_function(
            model=self.model_name,
            system=system_prompt,
            max_tokens=max_tokens,
            messages=other_messages,
            temperature=temperature,
            tools=tools,
            **think,
            **kwargs,
        )

        raw_response_dict = raw_response.to_dict()
        content = raw_response_dict['content']

        thinking_msgs, text_msgs, structured_msgs, tool_calls_raw = self._filter_messages(content)
        tool_calls = [ToolCall(self, tool_call) for tool_call in tool_calls_raw]

        message = {"role": "assistant", "content": raw_response.content}

        return LLMResponse(
            message=[message],
            content=text_msgs,
            thinking=thinking_msgs,
            raw_response=raw_response,
            structured_response=structured_msgs,
            tool_calls=tool_calls,
        )
                
    def get_num_tokens_response(self, response: LLMResponse):
        input_tokens = response.raw_response.usage.input_tokens
        output_tokens = response.raw_response.usage.output_tokens
        reasoning_tokens = None
        cached_tokens = response.raw_response.usage.cache_read_input_tokens
        return input_tokens, output_tokens, reasoning_tokens, cached_tokens


    @staticmethod
    def make_schema_for_tool(tool: Tool) -> dict:

        schema = {
            "name": tool.name,
            "description": tool.description,
        }
        
        if tool.arguments:
            schema['input_schema'] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
            }
            for arg in tool.arguments:
                schema['input_schema']['properties'][arg.name] = {
                    "type": arg.type,
                    "description": arg.description,
                }
                if arg.enum:
                    schema['input_schema']['properties'][arg.name]['enum'] = arg.enum
                if arg.items:
                    schema['input_schema']['properties'][arg.name]['items'] = {'type': arg.items}
                if arg.required:
                    schema['input_schema']['required'].append(arg.name)
        else:
            schema['input_schema'] = {
                "type": "object",
                "properties": {},
                "required": [],
            }

        return schema


    def generate_tool_response_message(
        self,
        tool_call : ToolCall,
    ) -> Dict[str, Any]:

        # parse response to string if not already
        if isinstance(tool_call.content, str):
            content = tool_call.content
        else:
            content = json.dumps(tool_call.content)

        return {
            'role': 'user',
            'content': [
                {
                    'tool_use_id' : tool_call.raw_tool_call['id'],
                    "type": "tool_result",
                    "content": content
                }
            ]
        }

    def get_tool_name(self, tool_call) -> str:
        return tool_call['name']

    def get_tool_args(self, tool_call) -> Dict[str, Any]:
        return tool_call['input']