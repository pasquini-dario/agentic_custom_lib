

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
        self.tool_methods = [
            getattr(self, attr) for attr in dir(self) 
            if callable(getattr(self, attr)) and hasattr(getattr(self, attr), '_is_tool')
        ]
        self.tools = [method() for method in self.tool_methods]
        self.tools_functions = {tool.name: tool.function for tool in self.tools}


    def print_tools(self):
        s = "🛠️ Available Tools:"
        s += "\n" + "=" * 60
        for tool in self.tools:
            s += "\n" + tool.print_tool()
        s += "\n" + "=" * 60
        print(s)