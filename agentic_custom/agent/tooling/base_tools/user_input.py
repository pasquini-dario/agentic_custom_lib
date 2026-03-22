from .. import Tool, Argument
from ..tools_context import tool

ASK_USER_TOOL_NAME = "ask_user"


def _build_ask_user_prompt(question: str) -> str:
    q = question.strip()
    return (
        "\n"
        "User input required\n"
        "\n"
        "The agent needs your answer to continue.\n"
        "\n"
        "Question:\n"
        f"{q}\n"
        "\n"
        "Your response: "
    )


@tool
def ask_user_for_input_tool(*args, **kwargs) -> Tool:
    """
    Tool to ask the user a question and return the answer.
    """
    def _ask_user(question: str) -> str:
        prompt = _build_ask_user_prompt(question)
        user_answer = input(prompt)
        return {"user_answer": user_answer, 'status': 'success'}

    return Tool(
        name=ASK_USER_TOOL_NAME,
        function=_ask_user,
        description="Ask the user a question and return the answer.",
        arguments=[
            Argument(
                name="question",
                description="The question to ask the user.",
                type="string",
            )
        ]
    )
