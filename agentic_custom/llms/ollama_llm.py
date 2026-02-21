import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ollama import Client
from httpx import ReadTimeout

from ..tooling import Tool, ToolCall
from . import LLM, LLMResponse, LLMTimeoutException

class OllamaLLM(LLM):

    SYSTEM_ROLE_NAME = 'developer'

    @staticmethod
    def check_requirements():
        return None
    
    def __init__(self, model_name: str, host=None, timeout=None):
        self.model_name = model_name
        self.host = host
        self.timeout = timeout
        self.client = Client(
            host=self.host,
            timeout=self.timeout,
        )
    
    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float= 0,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        format: Optional[Any] = None,
        think: bool = True,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
        ) -> LLMResponse:
        
        if format is not None:
            kwargs['format'] = format.model_json_schema()

        is_timeout = False
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'top_k': top_k,
                },
                tools=tools,
                think=think,
                **kwargs
            )
        except ReadTimeout:
            raise LLMTimeoutException()

        # extract thinking from response
        thinking = self.get_thinking_from_response(response)

        # Extract tool calls if present
        tool_calls = None
        if hasattr(response.message, 'tool_calls') and response.message.tool_calls:
            raw_tool_calls = response.message.tool_calls
            tool_calls = [ToolCall(self, raw_tool_call) for raw_tool_call in raw_tool_calls]

        # Handle structured format case
        parsing_error = None
        structured_response = None
        if format is not None:
            try:
                structured_response = format.model_validate_json(response.message.content).model_dump()
            except Exception as e:
                parsing_error = e
                
        response = LLMResponse(
            message=[response.message],
            content=response.message.content,
            tool_calls=tool_calls,
            thinking=thinking,
            structured_response=structured_response,
            raw_response=response,
            error=parsing_error,
        )
        return response
            

    def get_tool_name(self, tool_call: Dict[str, Any]) -> str:
        return tool_call.function.name  
    
    def get_tool_args(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        return tool_call.function.arguments

    def generate_tool_response_message(
        self,
        tool_call : ToolCall,
    ) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_name": tool_call.tool_name,
            "content": json.dumps(tool_call.content)
        }

    @staticmethod
    def make_schema_for_tool(tool: Tool) -> dict:

        schema = {
            'type': 'function',
            'function': {
                "name": tool.name,
                "description": tool.description,
            }
        }
        
        if tool.arguments:
            schema['function']['parameters'] = {
                    "type": "object",
                    "properties": {},
                    "required": [],
            }
            for arg in tool.arguments:
                schema['function']['parameters']['properties'][arg.name] = {
                    "type": arg.type,
                    "description": arg.description,
                }
                if arg.enum:
                    schema['function']['parameters']['properties'][arg.name]['enum'] = arg.enum
                if arg.items:
                    schema['function']['parameters']['properties'][arg.name]['items'] = {'type': arg.items}
                if arg.required:
                    schema['function']['parameters']['required'].append(arg.name)
        else:
            schema['function']['parameters'] = {}

        return schema

    def get_num_tokens_response(self, response: LLMResponse):
        input_tokens = response.raw_response.prompt_eval_count
        output_tokens = response.raw_response.eval_count
        print(f'[WARNING] missing reasoning and cached tokens numbers for Ollama LLM')
        return input_tokens, output_tokens, 0, 0

    def get_thinking_from_response(self, raw_response):
        return raw_response.message.get('thinking', None)