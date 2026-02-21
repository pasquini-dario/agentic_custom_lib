from dataclasses import dataclass
from typing import Any, Dict, List

from ..llms import LLMResponse
from ..tooling import ToolCall


@dataclass
class RoundPromise:
    """
    Class to store the output of a round of execution of the agent loop.
    A round is either:
    * A full iteation loop: an LLM inference and a SINGLE or NONE tool call execution.
    * If the LLM performs multiple tool invocations in the same round, a RoundPromise is created and returned for each.

    Important:
    * Currently, this class is not json-serializable.
    """

    TERMINATION_REASON_MAX_ITERATIONS = 'max_iterations'
    TERMINATION_REQUESTED = 'termination_requested'
    TASK_COMPLETED = 'task_completed'

    iteration: int = None
    messages_history: List[Dict[str, Any]] = None
    response: LLMResponse = None
    message: Dict[str, Any] = None
    tool_call: ToolCall = None
    termination: str = None

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'messages_history': self.messages_history,
            'response': self.response,
            'message': self.message,
            'tool_call': self.tool_call,
            'termination': self.termination,
        }
        
    def wait(self, timeout=None):
        """Wait for the tool call to be executed if asynchronous execution is enabled."""
        if self.tool_call is not None:
            self.tool_call.wait(timeout=timeout)

    def clone(self):
        return RoundPromise(**self.to_dict())

    def set_response(self, response: LLMResponse):
        self.response = response

    def set_messages_history(self, messages: List[Dict[str, Any]]):
        self.messages_history = messages

    def set_message(self, message: Dict[str, Any]):
        self.message = message

    def set_tool_call(self, tool_call: ToolCall):
        self.tool_call = tool_call

    def set_termination(self, termination: str):
        self.termination = termination

    def have_tools_been_called(self):
        return self.tool_call is not None