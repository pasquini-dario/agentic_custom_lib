import json
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import os

from .ollama_llm import OllamaLLM
from . import LLM, LLMResponse


class AzureLLM(OllamaLLM):
    HAS_COST = True

    @staticmethod
    def check_requirements():
        required_env_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'OPENAI_API_VERSION']
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                return f"{env_var} is not set"
        return None
    
    def __init__(self, model_name: str, **kargs):
        self.model_name = model_name
        self.client = AzureOpenAI(**kargs)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float=0,
        max_tokens: Optional[int] = None,
        format: Optional[BaseModel] = None,
        think: bool = True,
        tools: Optional[List[Dict[str, Any]]] = [],
        **kwargs
        ) -> LLMResponse:

        completion_kwargs = {}
        if format:
            completion_fun = self.client.beta.chat.completions.parse
            completion_kwargs['response_format'] = format

        else:
            completion_fun = self.client.chat.completions.create
            
        openai_response = completion_fun(
            model=self.model_name,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            tools=tools,
            messages=messages,
            reasoning_effort=think,
            **completion_kwargs,
        )

        if format:
            structured_response = openai_response.choices[0].message.parsed.model_dump()
        else:
            structured_response = None

        tool_calls = openai_response.choices[0].message.tool_calls
        if tool_calls is None:
            tool_calls = []

        return LLMResponse(
            message=[openai_response.choices[0].message],
            content=openai_response.choices[0].message.content,
            tool_calls=tool_calls,
            structured_response=structured_response,
            raw_response=openai_response
        )
            

    def generate_tool_response_message(
        self,
        fuction_call_item,
        tool_result,
    ) -> Dict[str, Any]:
        return {
            'tool_call_id' : fuction_call_item.id,
            "role": "tool",
            'name': self.get_tool_name(fuction_call_item),
            "content": json.dumps(tool_result.content)
        }

   
    def get_tool_name(self, tool_call) -> str:
        return tool_call.function.name

    def get_tool_args(self, tool_call) -> Dict[str, Any]:
        kargs = tool_call.function.arguments
        kargs = json.loads(kargs)
        return kargs

    def get_num_tokens_response(self, response: LLMResponse):
        input_tokens = response.raw_response.usage.prompt_tokens
        generated_tokens = response.raw_response.usage.completion_tokens
        output_tokens = generated_tokens
        return input_tokens, output_tokens

    