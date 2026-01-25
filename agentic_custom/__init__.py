from .tooling.tools_context import ToolsContext, tool 
from .tooling import Tool, Argument, ToolResult
from .agent import Agent, AgentTerminationException
from .run_tracker import LLMRunTracker
from .llms import LLM, LLMTimeoutException


from .llms.ollama_llm import OllamaLLM
from .llms.azure_llm import AzureLLM
from .llms.openai_llm import OpenaiLLM

PROVIDERS_MAP = {
    "ollama": OllamaLLM,
    "azure": AzureLLM,
    'openai': OpenaiLLM,
}

def check_llm_provider_requirements(provider_name: str) -> str:
    """Checks if the necessary requirements are met for the given LLM provider."""
    if provider_name not in PROVIDERS_MAP:
        return f"Provider {provider_name} not found"
    return PROVIDERS_MAP[provider_name].check_requirements()

def load_llm(provider_name: str, model_name: str, *args, **kwargs):
    """Loads an LLM from the given provider and model name."""
    if provider_name not in PROVIDERS_MAP:
        raise ValueError(f"Provider {provider_name} not found")
    return PROVIDERS_MAP[provider_name](model_name, *args, **kwargs)

__all__ = [
    "ToolsContext",
    "Tool",
    "Argument",
    "ToolResult",
    "Agent",
    "AgentTerminationException",
    "LLMRunTracker",
    "LLM",
    "check_llm_provider_requirements",
    "load_llm",
    "LLMTimeoutException",
]