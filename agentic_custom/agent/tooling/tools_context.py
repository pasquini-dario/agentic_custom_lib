from __future__ import annotations

import copy
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from agentic_custom.agent import Agent

def tool(func):
    """
    Decorator to mark a method as a tool method.
    Methods decorated with @tool will be automatically discovered and registered.
    """
    func._is_tool = True
    return func

class ToolsContext:
    """
    Class the manages contextual tool creation and execution.
    To define a new tool, you need to decorate a method with @tool.
    The function should return a Tool object.
    For example:
    ```
    @tool
    def modify_current_candidate_via_python_code(self) -> str:
    
        def fun(code: str) -> str:
            try:
                ns = {}
                exec(code, ns)
                new_content = ns['modify_current_candidate'](self.candidate_file.content)
                self.candidate_file.update_file(new_content)
                return "Modification successful"
            except Exception as e:
                return f"Modification failed: {e}"

            tool = Tool(
                name='modify_current_candidate_via_python_code',
                function=fun,
                description="Modify the current candidate file via a Python code. It returns a string "Modification successful" if the modification is successful, a string describing the error otherwise.",
                arguments=[
                    Argument(
                        name='code',
                        description='A string containing Python code implementing a function defined as: def modify_current_candidate(current_candidate_path: bytes) -> bytes',
                        type='string')
                    ]
            )
            return tool
    ```

    When the object is initialized, it will create a dictionary with the tools names as keys and the functions as values.
    """
    def __init__(self, *args, **kwargs):
        # collect decorated tools from the class
        self.tools = [
            getattr(self, attr)() for attr in dir(self) 
            if callable(getattr(self, attr)) and hasattr(getattr(self, attr), '_is_tool')
        ]
        # setup tools dictionary
        self.setup_tools()
        self.associated_agent = None

    def setup_tools(self):
        self.tools_functions = {tool.name: tool.function for tool in self.tools}
        
    def get_tool_names(self) -> List[str]:
        return [tool.name for tool in self.tools]

    def print_tools(self):
        s = "🛠️ Available Tools:"
        s += "\n" + "=" * 60
        for tool in self.tools:
            s += "\n" + tool.print_tool()
        s += "\n" + "=" * 60
        print(s)

    def add_tools(self, tool_context_to_add, tool_names_to_add: List[str]=None):
        """
        Add tools from another tools context to the current tools context.
        If tool_names_to_add is provided, only the tools with the given names will be added.
        If tool_names_to_add is not provided, all tools will be added.
        """
        tools_to_add = tool_context_to_add.tools
        if tool_names_to_add is not None:
            tools_to_add = filter(lambda tool: tool.name in tool_names_to_add, tools_to_add)

        if not tools_to_add:
            raise ValueError(f"No tools to add from {tool_context_to_add}. Double check the tool names.")

        self.tools.extend(tools_to_add)
        self.setup_tools()

    def register_to_agent(self, agent: Agent) -> ToolsContext:
        """
        Creates a shallow copy of this ToolsContext bound to a specific Agent.

        The returned copy has its own `associated_agent` attribute set to `agent`,
        while all other attributes (e.g. `tools`, `tools_functions`, and attributes added by subclasses) remain shared
        references with the original instance.
        """
        cloned = copy.copy(self)
        cloned.associated_agent = agent
        return cloned


