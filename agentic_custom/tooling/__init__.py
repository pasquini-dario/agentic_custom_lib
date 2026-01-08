from dataclasses import dataclass
from typing import Callable, List, Any

class Argument:
    def __init__(
        self,
        name: str,
        description: str,
        type: str,
        enum: List[str] = None,
        items: str = None,
        required: bool = True,
    ):
        self.name = name
        self.description = description
        self.type = type
        if type == 'array':
            if items is None:
                raise ValueError("items is required for array type")
        self.enum = enum
        self.items = items
        self.required = required

    def print_argument(self):
        required_emoji = "✅" if self.required else "❌"
        enum_str = ""
        if self.enum:
            enum_values = ', '.join(map(str, self.enum))
            enum_str = f"     ├─ 🔢 Enum: [{enum_values}]\n"
        return (
            f"  └─ ⚙️  Argument: {self.name}\n"
            f"     ├─ 📝 Description: {self.description}\n"
            f"     ├─ 🎯 Type: {self.type}\n"
            f"{enum_str}"
            f"     └─ {required_emoji} Required: {self.required}"
        )

class Tool:
    """
     An helper class used to define a tool. Used for automated schema generation based on the LLM in used.
    """
    def __init__(
        self,
        name: str,
        function: Callable,
        description: str,
        arguments: List[Argument] = [],
    ):
        self.name = name
        self.function = function
        self.description = description
        self.arguments = arguments

    def print_tool(self):
        lines = [
            "=" * 60,
            f"🔧 Tool: {self.name}",
            f"📝 Description: {self.description}",
        ]
        if self.arguments:
            lines.append(f"📋 Arguments ({len(self.arguments)}):")
            for argument in self.arguments:
                lines.append(argument.print_argument())
        else:
            lines.append("📋 Arguments: None")
        lines.append("=" * 60)
        return "\n".join(lines)


class ToolResult:
    """
    A class to wrap the result of a tool invocation. 
    """
    def __init__(self, is_tool_invocation_successful: bool = True, content: Any = None, is_termination: bool = False):
        self.is_tool_invocation_successful = is_tool_invocation_successful
        self.content = content
        self.is_termination = is_termination