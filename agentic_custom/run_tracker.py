from collections import defaultdict
from pprint import pformat


from .llms import LLM, LLMResponse
from .cost import cost_calculator
from .tooling import ToolResult

class LLMRunTracker:
    
    def __init__(self, llm: LLM):
        self.llm = llm
        self.tot_input_tokens = 0
        self.tot_output_tokens = 0
        self.cached_tokens = 0
        self.num_messages = 0
        self.tool_invocation_counts = defaultdict(int)

        self.messages = []

        self.total_cost = 0

    def set_llm(self, llm: LLM):
        self.llm = llm
        print(f"RunTracker: LLM set to {self.llm.model_name}")

    def print_tool_invocation(self,tool_call):
        print(f"{'=' * 70}")
        name = self.llm.get_tool_name(tool_call)
        args = self.llm.get_tool_args(tool_call)
        print(f"┌────────────────────────── 🛠️ Tool Call ──────────────────────────")
        print(f"│ Name      : {name}")
        print(f"│ Arguments :")
        if args:
            args_str = pformat(args, indent=2, width=60)
            args_lines = args_str.splitlines()
            for line in args_lines:
                print(f"│   {line}")
        else:
            print(f"│   (No arguments)")
        print(f"└─────────────────────────────────────────────────────────────────────\n")

    def print_message(self, llm_response):
        print(f"{'=' * 70}")
        print(f"┌────────────────────────── Message ──────────────────────────")
        thinking = llm_response.thinking
        if thinking:
            print(f"│ 💡 Thinking  :")
            thinking_str = pformat(thinking, indent=2, width=60)
            thinking_lines = thinking_str.splitlines()
            for line in thinking_lines:
                print(f"│   {line}")
        content = llm_response.content
        print(f"│ 📄 Content   :")
        if content:
            content_str = pformat(content, indent=2, width=60)
            content_lines = content_str.splitlines()
            for line in content_lines:
                print(f"│   {line}")
        else:
            print(f"│   (No content)")
        message = llm_response.message
        print(f"│ Message   :")
        message_str = pformat(message, indent=2, width=60)
        message_lines = message_str.splitlines()
        for line in message_lines:
            print(f"│   {line}")
        print(f"└─────────────────────────────────────────────────────────────────────\n")

    def print_tool_result(self, tool_result: ToolResult):
        print(f"{'=' * 70}")
        # Extract fields from tool_result (typically a dict with tool_call_id, role, name, content)
        content = tool_result.content
        
        if not tool_result.is_tool_invocation_successful:
            print(f"┌────────────────────────── ❌ Tool Error ──────────────────────────")
            print(f"│ Error       :")
            if content:
                content_str = pformat(content, indent=2, width=60)
                content_lines = content_str.splitlines()
                for line in content_lines:
                    print(f"│   {line}")
            else:
                print(f"│   (No error details)")
            print(f"└─────────────────────────────────────────────────────────────────────\n")
        else:
            print(f"┌────────────────────────── ✅ Tool Result ──────────────────────────")
            print(f"│ Result       :")
            if content:
                content_str = pformat(content, indent=2, width=60)
                content_lines = content_str.splitlines()
                for line in content_lines:
                    print(f"│   {line}")
            else:
                print(f"│   (No result)")
            print(f"└─────────────────────────────────────────────────────────────────────\n")

    def print_termination(self, reason):
        print(f"{'=' * 70}")
        print(f"┌────────────────────────── 🛑 Termination ──────────────────────────")
        print(f"│ Reason     :")
        reason_str = pformat(reason, indent=2, width=60)
        reason_lines = reason_str.splitlines()
        for line in reason_lines:
            print(f"│   {line}")
        print(f"└─────────────────────────────────────────────────────────────────────\n")

    def add_message(self, llm_response: LLMResponse, verbose=False):
        if llm_response.is_successful():
            input_tokens, output_tokens = self.llm.get_num_tokens_response(llm_response)
            if input_tokens:
                self.tot_input_tokens += input_tokens
            if output_tokens:
                self.tot_output_tokens += output_tokens

            if self.llm.HAS_COST:
                cost = cost_calculator(self.llm.model_name, input_tokens, output_tokens)
                if cost is not None:
                    self.total_cost += cost

            self.num_messages += 1
            self.messages.append(llm_response.message)

        if verbose:
            if llm_response.is_successful():
                if llm_response.tool_calls:
                    print(f'#{len(llm_response.tool_calls)} Tool calls')
                else:
                    self.print_message(llm_response)
            else:
                print(f'[ERROR] --> {llm_response.error}')
    
    def add_tool_invocation(self, tool_invocation, verbose):

        if verbose:
            self.print_tool_invocation(tool_invocation)
       
        name = self.llm.get_tool_name(tool_invocation)
        self.tool_invocation_counts[name] += 1


    def add_tool_result(self, tool_result, verbose):
        if verbose:
            self.print_tool_result(tool_result)


    def signal_termination(self, reason, verbose):
        if verbose:
            self.print_termination(reason)


    def print_summary(self):
        print(f"{'=' * 70}")
        print(f'# Summary #########################################################')
        print(f'Total messages: {self.num_messages}')
        print(f'Total input tokens: {self.tot_input_tokens}')
        print(f'Total output tokens: {self.tot_output_tokens}')
        if self.llm.HAS_COST:
            print(f'Total cost: {self.total_cost:.4f} USD')
        else:
            print(f'Total cost: N/A')
        print('Tool invocation counts:')
        if self.tool_invocation_counts:
            for tool_name, count in self.tool_invocation_counts.items():
                print(f'  - {tool_name}: {count}')
        else:
            print('  (none)')
        print(f'########################################################')