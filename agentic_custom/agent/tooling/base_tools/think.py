from .. import Tool, Argument
from ..tools_context import tool

THINK_TOOL_NAME = "think"

_DEFAULT_THINK_DESCRIPTION = "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."
_DEFAULT_THINK_ARGUMENT_DESCRIPTION = "The thought to think about. It will be appended to the log."

def make_think_tool(description: str, argument_description: str) -> Tool:
    @tool
    def think_tool(*args, **kwargs) -> Tool:
        """
        Explict thinking via tool https://www.anthropic.com/engineering/claude-think-tool
        """
        def _think(thought: str) -> str:
            return thought

        return Tool(
            name=THINK_TOOL_NAME,
            function=_think,
            description=description,
            arguments=[
                Argument(
                    name="thought",
                    description=argument_description,
                    type="string",
                )
            ]
        )
    return think_tool


think_tool = make_think_tool(description=_DEFAULT_THINK_DESCRIPTION, argument_description=_DEFAULT_THINK_ARGUMENT_DESCRIPTION)
