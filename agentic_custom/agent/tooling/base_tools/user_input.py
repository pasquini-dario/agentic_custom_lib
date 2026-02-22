from .. import Tool, Argument
from ..tools_context import tool

_USER_PROMPT = """
╔══════════════════════════════════════════════════════════════════╗
║                      🤖 USER INPUT REQUIRED                      ║
╠══════════════════════════════════════════════════════════════════╣
║  The AI agent needs your input to proceed.                       ║
║                                                                  ║
║  Question: {question}
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
➤ Your response: """

@tool
def ask_user_for_input_tool(*args, **kwargs) -> Tool:
    """
    Tool to ask the user a question and return the answer.
    """
    def _ask_user(question: str) -> str:
        prompt = _USER_PROMPT.format(question=question)
        user_answer = input(prompt)
        return {"user_answer": user_answer, 'status': 'success'}

    return Tool(
        name="ask_user",
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
