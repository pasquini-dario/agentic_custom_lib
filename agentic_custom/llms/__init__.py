from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel
from types import SimpleNamespace

from ..tooling import Tool

class LLMTimeoutException(Exception):
    pass

class LLMResponse(SimpleNamespace):
    """ Class for LLM responses """

    @staticmethod
    def check_requirements():
        """
            This method checks if the necessary requirements are met for the LLM to function correctly.
            This includes checking if API keys are set or the necessary services are available.
            It should be implemented for each LLM class.

            Returns:
                str: None if the requirements are met, otherwise a message indicating the missing requirements
        """


    def __init__(
        self, 
        message: Any,
        content: str,
        tool_calls: List[Dict[str, Any]] = [],
        thinking: Optional[str] = None,
        raw_response: Optional[Any] = None,
        structured_response: Optional[Any] = None,
        error: Optional[Exception] = None,
    ):
        super().__init__()
        error = self.check_requirements()
        if error:
            raise Exception(error)
        self.message = message
        self.content = content
        self.tool_calls = tool_calls
        self.thinking = thinking
        self.raw_response = raw_response
        self.structured_response = structured_response
        self.error = error

    def __getitem__(self, key):
        return getattr(self, key, None)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def is_successful(self) -> bool:
        """
        Check if the LLM generation was successful.
        
        Returns:
            bool: True if no error occurred, False otherwise
        """
        return self.error is None
    
    def has_tool_calls(self) -> bool:
        """
        Check if tools have been called in the response.
        
        Returns:
            bool: True if tool_calls exist and are not empty, False otherwise
        """
        return self.tool_calls is not None and len(self.tool_calls) > 0
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get the list of tool calls if any.
        
        Returns:
            List[Dict[str, Any]]: List of tool calls, empty list if none
        """
        return self.tool_calls if self.tool_calls is not None else []


class LLM:
    HAS_COST = False
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float= 0,
        max_tokens: Optional[int] = None,
        top_k: Optional[int] = None,
        format: Optional[BaseModel] = None,
        think: bool = True,
        **kwargs):
        raise NotImplementedError("This method is not implemented for this LLM")

    def get_tool_name(self, tool_call) -> str:
        raise NotImplementedError("This method is not implemented for this LLM")

    def get_tool_args(self, tool_call) -> Dict[str, Any]:
        raise NotImplementedError("This method is not implemented for this LLM")

    def generate_tool_response_message(
        self,
        tool_call,
        function_output,
    ) -> Dict[str, Any]:
        raise NotImplementedError("This method is not implemented for this LLM")

    def get_num_tokens_response(self, response: LLMResponse) -> Tuple[int, int]:
        print('[WARNING] LLM.get_num_tokens_response method is not implemented for this LLM')
        return 0, 0

    def make_schema_for_tool(tool: Tool) -> dict:
        raise NotImplementedError("This method is not implemented for this LLM")

    def get_thinking_from_response(self, response: LLMResponse) -> str:
        return None
