from pprint import pformat
from typing import Protocol

from .llms import LLMResponse
from .agent.tooling import ToolCall


class _LLMSummaryView(Protocol):
    HAS_COST: bool


class SummaryView(Protocol):
    num_messages: int
    tot_input_tokens: int
    tot_output_tokens: int
    tot_reasoning_tokens: int
    tot_cached_tokens: int
    total_cost: dict
    tool_invocation_counts: dict
    llm: _LLMSummaryView

    def get_cached_tokens_percentage(self) -> float: ...


class RunVisualizer:

    def print_llm_set(self, model_name: str):
        print(f"RunTracker: LLM set to {model_name}")

    def print_tool_invocation(self, tool_call: ToolCall):
        print(f"{'=' * 70}")
        name = tool_call.tool_name
        args = tool_call.tool_args
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

    def print_message(self, llm_response: LLMResponse):
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

    def print_tool_result(self, tool_call: ToolCall):
        print(f"{'=' * 70}")
        content = tool_call.content

        if not tool_call.is_tool_invocation_successful:
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

    def get_summary(self, tracker: SummaryView, default_context_key: str) -> str:
        lines = [
            f"{'=' * 70}",
            f'# Summary #########################################################',
            f'Total messages: {tracker.num_messages}',
            f'Total input tokens: {tracker.tot_input_tokens}',
            f'Total output tokens: {tracker.tot_output_tokens}',
            f'Total reasoning tokens: {tracker.tot_reasoning_tokens}',
            f'Total cached tokens: {tracker.tot_cached_tokens}',
            f'Total uncached output tokens: {tracker.tot_input_tokens - tracker.tot_cached_tokens}',
            f'Cached tokens hit ratio: {tracker.get_cached_tokens_percentage() * 100:.2f}%',
        ]
        if tracker.llm.HAS_COST:
            if len(tracker.total_cost) == 1 and default_context_key in tracker.total_cost:
                lines.append(f'Total cost: {tracker.total_cost[default_context_key]:.4f} USD')
            else:
                for context_key, cost in tracker.total_cost.items():
                    lines.append(f'  - {context_key}: {cost:.4f} USD')
                total = sum(tracker.total_cost.values())
                lines.append(f'Total cost: {total:.4f} USD')
        else:
            lines.append('Total cost: N/A')
        lines.append('Tool invocation counts:')
        if tracker.tool_invocation_counts:
            for tool_name, count in tracker.tool_invocation_counts.items():
                lines.append(f'  - {tool_name}: {count}')
        else:
            lines.append('  (none)')
        lines.append('########################################################')
        return '\n'.join(lines)
