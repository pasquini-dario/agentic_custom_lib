from enum import Enum
from typing import Dict, Any, List, Callable

from .llms import LLM
from .run_tracker import LLMRunTracker
from .tooling.tools_context import ToolsContext
from .tooling import ToolResult

class AgentTerminationException(Exception):
    pass

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
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_data}
        ]

    def single_execution(self, user_data: str, **kargs):
        messages = self.get_ancestor_messages(user_data)
        return self.execute(messages, **kargs)

    def generate_tool_schemas(self, tools_context: ToolsContext):
        return [self.llm.make_schema_for_tool(tool) for tool in tools_context.tools]

    def execute_agent_loop(
        self,
        input_args,
        tools_context: ToolsContext,
        get_response_hook: Callable=None,
        after_tool_execution_hook: Callable=None,
        messages: List[Dict[str, Any]]=None,
        verbose=True,
    ):
        if messages is None:
            # get initial messages
            messages = self.get_ancestor_messages(*input_args)
        # list schemas for tools
        tool_schemas = self.generate_tool_schemas(tools_context)
        # execute agent loop
        for i in range(self.max_iterations):
            response = self.execute(messages, tools=tool_schemas)
            # log message for stats tracking
            self.run_tracker.add_message(response, verbose)
            message = response.message
            # update history with model answer
            messages += message

            if get_response_hook:
                get_response_hook(response, messages)

            # if tool calls are present, execute tools
            if response.tool_calls:
                for toll_call in response.tool_calls:
                    # log tool invocation for stats tracking
                    self.run_tracker.add_tool_invocation(toll_call, verbose)
                    # execute tool
                    tool_result = self.execute_tool(toll_call, tools_context)
                    tool_message = self.llm.generate_tool_response_message(toll_call, tool_result)
                    
                    if tool_result.is_termination:
                        # Termination requested
                        self.run_tracker.signal_termination('Termination requested', verbose)
                        return messages

                    # log tool result for stats tracking
                    self.run_tracker.add_tool_result(tool_result, verbose)
                    
                    # update history with tool result
                    messages.append(tool_message)  

                    if after_tool_execution_hook:
                        after_tool_execution_hook(toll_call, tool_message)
        else:
            self.run_tracker.signal_termination('Max iterations reached', verbose)

                    
        return messages

    def execute_tool(
        self,
        tool_call,
        tools_context: ToolsContext
    ):
        name = self.llm.get_tool_name(tool_call)
        args = self.llm.get_tool_args(tool_call)
        try:
            content = tools_context.tools_functions[name](**args)
            result = ToolResult(is_tool_invocation_successful=True, content=content)    
        except AgentTerminationException as ex:
            result = ToolResult(is_termination=True)
        except Exception as ex:
            result = ToolResult(is_tool_invocation_successful=False, content=str(ex))
        return result