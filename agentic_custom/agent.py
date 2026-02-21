from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass, replace, asdict

from .llms import LLM, LLMResponse
from .run_tracker import LLMRunTracker
from .tooling import ToolCall
from .tooling.tools_context import ToolsContext
from .run_tracker import DEFAULT_CONTEXT_KEY


# Utility to create Enum dynamically
def create_enum(name: str, values: List[str]) -> Enum:
    return Enum(name, {v.replace(" ", "_").replace("-", "_").lower(): v for v in values})

class Agent:
    output_model = None
    system_prompt = None

    def __init__(
        self,
            llm: LLM,
            max_iterations: int=2**8,
            generation_params: Dict[str, Any]={},
            run_tracker: LLMRunTracker=None,
        ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.generation_params = generation_params
        if run_tracker is None:
            self.run_tracker = LLMRunTracker(self.llm)
        else:
            self.run_tracker = run_tracker
            self.run_tracker.set_llm(self.llm)

    def execute(self, messages: List[Dict[str, Any]], **kargs):
        # Use configuration parameters if not overridden in kargs
        generation_params = self.generation_params.copy()
        
        # Override with any parameters passed in kargs
        generation_params.update(kargs)

        return self.llm.generate(
            messages,
            format=self.output_model,
            **generation_params
        )
         
    def get_ancestor_messages(self, user_data: str):
        messages = [
            {'role': self.llm.SYSTEM_ROLE_NAME, 'content': self.system_prompt},
        ]
        if user_data:
            messages.append({'role': 'user', 'content': user_data})
        return messages

    def single_execution(self, user_data, context_key=DEFAULT_CONTEXT_KEY, **kargs):
        messages = self.get_ancestor_messages(*user_data)
        response = self.execute(messages, **kargs)
        self.run_tracker.add_message(response, context_key=context_key)
        return response

    def generate_tool_schemas(self, tools_context: ToolsContext, enabled_tools_keys: List[str]=None):
        if enabled_tools_keys is not None:
            tools_context.tools = [tool for tool in tools_context.tools if tool.name in enabled_tools_keys]
        return [self.llm.make_schema_for_tool(tool) for tool in tools_context.tools]

    def execute_agent_loop(
        self,
        input_args,
        tools_context: ToolsContext,
        messages: List[Dict[str, Any]]=None,
        enabled_tools_keys: List[str]=None,
        verbose=True,
        context_key=DEFAULT_CONTEXT_KEY,
    ):
        """ 
            Execute the agent loop, given a ToolsContext object defining the tools to be used.
            if enabled_tools_keys is not None, only the tools with the keys in the list will be enabled. If it is None, all tools will be enabled.
        """
        if messages is None:
            # get initial messages
            messages = self.get_ancestor_messages(*input_args)
        # list schemas for tools
        tool_schemas = self.generate_tool_schemas(tools_context, enabled_tools_keys=enabled_tools_keys)
        # execute agent loop
        for i in range(self.max_iterations):

            round_output = RoundOutput(iteration=i)
            
            # raw response from the model
            response = self.execute(messages, tools=tool_schemas)
            round_output.set_response(response)
            self.run_tracker.add_message(response, verbose, context_key=context_key)

            # log message for stats tracking
            message = response.message
            round_output.set_message(message)
            # update history with model answer
            messages += message

            self._get_response_hook(i, response, messages)

            # if tool calls are present, execute tools
            terminated = False
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    # clone round output to avoid modifying the original object as multiple tool calls may be executed in the same round starting from the same LLM response
                    round_output = round_output.clone()

                    # log tool invocation for stats tracking
                    round_output.set_tool_call(tool_call)
                    self.run_tracker.add_tool_invocation(tool_call, verbose)
                    # execute tool
                    tool_call.run_sync(tools_context)

                    round_output.set_messages_history(messages)
                    # return round output to the caller
                    yield round_output

                    if not tool_call.is_executed():
                        # Make sure the tool call has been executed in case of an asynchronous tool call.
                        raise Exception(f"Tool {tool_call.tool_name} has not been executed. Async tool calls must be handled by the developer. Use the wait() method to wait for the tool call to be executed before continuing the agentic loop.")

                    tool_message = tool_call.generate_tool_response_message()
                    # update history with tool result
                    messages.append(tool_message)  
                    
                    # log tool result for stats tracking
                    self.run_tracker.add_tool_result(tool_call, verbose)
                    self._after_tool_execution_hook(i, tool_call, tool_message)

                    if tool_call.is_termination:
                        # Termination requested
                        self.run_tracker.signal_termination('Termination requested', verbose)
                        round_output.set_termination(RoundOutput.TERMINATION_REQUESTED)
                        terminated = True

            else:
                # if no tool calls
                round_output.set_messages_history(messages)
                yield round_output


            if terminated:
                # explicit exit-condition met
                return

            # external hook to transform messages (default behavior is to return the messages as is)
            messages = self._end_round_messages_transformation_hook(i, messages)
        else:
            # max iterations reached
            self.run_tracker.signal_termination('Max iterations reached', verbose)
            round_output.set_termination(RoundOutput.TERMINATION_REASON_MAX_ITERATIONS)
            round_output.set_messages_history(messages)
            yield round_output


    # Hook functions
    def _get_response_hook(self, round_number: int, response: LLMResponse, messages: List[Dict[str, Any]]):
        """ Hook function called after the model response is generated """
        ...

    def _after_tool_execution_hook(self, round_number: int, tool_call: Dict[str, Any], tool_message: Dict[str, Any]):
        """ Hook function called after a tool is executed """
        ...

    def _end_round_messages_transformation_hook(self, round_number: int, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Hook function called after the round is ended. The value returned is the messages to be used in the next round. This can be used for message filtering and pruning."""
        return messages
    #########################################################################################


@dataclass
class RoundOutput:
    """
    Class to store the output of a round of execution of the agent loop.
    A round is either:
    * A full iteation loop: an LLM inference and a SINGLE or NONE tool call execution.
    * If the LLM performs multiple tool invocations in the same round, a RoundOutput is created and returned for each.

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

    # def to_dict(self):
    #     return asdict(self)

    # def clone(self):
    #     return replace(self)

    def to_dict(self):
        return {
            'iteration': self.iteration,
            'messages_history': self.messages_history,
            'response': self.response,
            'message': self.message,
            'tool_call': self.tool_call,
            'termination': self.termination,
        }
        
    def wait(self):
        """ Wait for the tool call to be executed if asynchronous execution is enabled."""
        raise NotImplementedError("")

    def clone(self):
        return RoundOutput(**self.to_dict())

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
        return not self.tool_result is None
