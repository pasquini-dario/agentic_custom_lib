from collections import defaultdict

from .llms import LLM, LLMResponse
from .cost import cost_calculator
from .agent.tooling import ToolCall
from .run_visualize import RunVisualizer

DEFAULT_CONTEXT_KEY = 'default'

class LLMRunTracker:
    """
    Class for tracking the execution of an agent:
    - keep track of the number of tokens used by the LLM and the cost
    - logs LLM responses and tool calls
    """

    def __init__(self, llm: LLM):
        self.llm = llm
        self._visualizer = RunVisualizer()
        self.tot_input_tokens = 0
        self.tot_output_tokens = 0
        self.tot_reasoning_tokens = 0
        self.tot_cached_tokens = 0
        self.num_messages = 0

        self.tool_invocation_counts = defaultdict(int)

        self.messages = defaultdict(list)
        self.tool_calls = defaultdict(list)
        self.total_cost = defaultdict(int)

    def set_llm(self, llm: LLM):
        self.llm = llm
        self._visualizer.print_llm_set(self.llm.model_name)

    def add_message(self, llm_response: LLMResponse, verbose=False, context_key=DEFAULT_CONTEXT_KEY):
        if llm_response.is_successful():
            input_tokens, output_tokens, reasoning_tokens, cached_tokens = self.llm.get_num_tokens_response(llm_response)
            if input_tokens:
                self.tot_input_tokens += input_tokens
            if output_tokens:
                self.tot_output_tokens += output_tokens
            if reasoning_tokens:
                self.tot_reasoning_tokens += reasoning_tokens
            if cached_tokens:
                self.tot_cached_tokens += cached_tokens

            if self.llm.HAS_COST:
                cost = cost_calculator(self.llm.model_name, input_tokens, output_tokens, cached_tokens)
                if cost is not None:
                    self.total_cost[context_key] += cost

            self.num_messages += 1
            self.messages[context_key].append(llm_response)

        if verbose:
            if llm_response.is_successful():
                if llm_response.tool_calls:
                    print(f'#{len(llm_response.tool_calls)} Tool calls')
                else:
                    self._visualizer.print_message(llm_response)
            else:
                print(f'[ERROR] --> {llm_response.error}')

    def add_tool_invocation(self, tool_call: ToolCall, verbose=False, context_key:str=DEFAULT_CONTEXT_KEY):
        if verbose:
            self._visualizer.print_tool_invocation(tool_call)

        name = tool_call.tool_name
        self.tool_invocation_counts[name] += 1
        self.tool_calls[context_key].append(tool_call)

    def add_tool_result(self, tool_call: ToolCall, verbose):
        if verbose:
            self._visualizer.print_tool_result(tool_call)

    def signal_termination(self, reason, verbose):
        if verbose:
            self._visualizer.print_termination(reason)

    def get_cached_tokens_percentage(self):
        if self.tot_input_tokens == 0:
            return 0.0
        return self.tot_cached_tokens / self.tot_input_tokens

    def print_summary(self):
        print(self.get_summary())

    def get_summary(self):
        return self._visualizer.get_summary(self, DEFAULT_CONTEXT_KEY)