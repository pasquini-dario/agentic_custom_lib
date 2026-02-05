from openai import OpenAI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import json
import os

from . import LLM, LLMResponse
from ..tooling import Argument, Tool

class OpenaiLLM(LLM):

    HAS_COST = True
    SYSTEM_ROLE_NAME = 'developer'

    @staticmethod
    def check_requirements():
        if not os.getenv('OPENAI_API_KEY'):
            return "OPENAI_API_KEY is not set"
        return None

    def __init__(self, model_name: str, **kargs):
        self.model_name = model_name
        self.client = OpenAI(**kargs)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float=None,
        max_tokens: Optional[int] = None,
        format: Optional[BaseModel] = None,
        think: Any = None,
        tools: Optional[List[Dict[str, Any]]] = [],
        **kwargs
        ) -> LLMResponse:

        if format is None:
            gen_function = self.client.responses.create
            structured_output = None
        else:
            gen_function = self.client.responses.parse
            kwargs['text_format'] = format

        if isinstance(think, str):
            think = {'effort' : think}


        openai_response = gen_function(
            model=self.model_name,
            input=messages,
            reasoning=think,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=tools,
            **kwargs,
        )

        if not format is None:
            structured_output = openai_response.output_parsed.model_dump()

        # function calling
        tool_calls = [output for output in openai_response.output if output.type == 'function_call']

        response = LLMResponse(
            message=openai_response.output,
            content=openai_response.output_text,
            tool_calls=tool_calls,
            thinking=None,
            raw_response=openai_response,
            structured_response=structured_output,
            error=None,
        )

        return response


    @staticmethod
    def make_schema_for_tool(tool: Tool) -> dict:

        schema = {
            'type': 'function',
            "name": tool.name,
            "description": tool.description,
        }
        
        if tool.arguments:
            schema['parameters'] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
            }
            for arg in tool.arguments:
                schema['parameters']['properties'][arg.name] = {
                    "type": arg.type,
                    "description": arg.description,
                }
                if arg.enum:
                    schema['parameters']['properties'][arg.name]['enum'] = arg.enum
                if arg.items:
                    schema['parameters']['properties'][arg.name]['items'] = {'type': arg.items}
                if arg.required:
                    schema['parameters']['required'].append(arg.name)
        else:
            schema['parameters'] = {}

        return schema

    def generate_tool_response_message(
        self,
        tool_call_item,
        tool_result,
    ) -> Dict[str, Any]:
        return {
            "type": "function_call_output",
            "call_id": tool_call_item.call_id,
            "output": json.dumps(tool_result.content),
        }

    def get_tool_name(self, tool_call) -> str:
        return tool_call.name

    def get_tool_args(self, tool_call) -> Dict[str, Any]:
        kargs = tool_call.arguments
        kargs = json.loads(kargs)
        return kargs

    def get_num_tokens_response(self, response: LLMResponse):
        input_tokens = response.raw_response.usage.input_tokens
        output_tokens = response.raw_response.usage.output_tokens
        reasoning_tokens = response.raw_response.usage.output_tokens_details.reasoning_tokens
        cached_tokens = response.raw_response.usage.input_tokens_details.cached_tokens
        return input_tokens, output_tokens, reasoning_tokens, cached_tokens