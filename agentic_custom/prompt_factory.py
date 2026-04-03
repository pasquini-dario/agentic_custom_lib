
from typing import Any, Dict, List, Callable, Optional

from abc import ABC
from typing import Any, ClassVar, Generic, TypeVar, Unpack

from jinja2 import StrictUndefined, Template


# class PromptFactory2:
#     """
#     Abstract base for constructing provider-aware LLM prompts from modular sections.

#     Subclasses must define:
#     - _ROOT_PROMPT: a Jinja2 template string with {{ placeholder }} references to
#       section names (and optionally to call-time variables that override sections).

#     Optional (default to {} if absent):
#     - _sections: dict of shared/default prompt sections (provider-agnostic).
#     - One dict per provider (_openai_sections, _anthropic_sections, etc.) that overrides
#       or extends _sections for that provider.
#     - _metadata: dict of shared/default metadata (e.g. cache_control) for the prompt content block.
#     - One dict per provider (_openai_metadata, _anthropic_metadata, etc.) that overrides
#       or extends _metadata for that provider. Values are ready-to-use in the API wire format.

#     Section values and _ROOT_PROMPT are compiled as Jinja2 templates (StrictUndefined).
#     At call time, _sections is merged with the selected provider dict (provider wins on
#     conflict). Each section template is rendered with get_content(provider, **kwargs).
#     Rendered section names are then merged with kwargs (kwargs win), and the result
#     is used to render _ROOT_PROMPT. So call-time arguments can override section
#     outputs when filling the root template.

#     The generic parameter _KwargsT is a TypedDict that types the **kwargs of get_content,
#     so subclasses get typed kwargs without needing to override get_content.

#     Example:
#     ```python
#     class _MemoryParams(TypedDict):
#         user_name: str

#     class MemorySystemPrompt(BasePrompt[_MemoryParams]):
#         _ROOT_PROMPT = "{{ memory_role }}\n{{ memory_guidelines }}\n{{ memory_format }}"

#         _sections = {
#             "memory_role": "You are a memory agent for {{ user_name }}.",
#             "memory_guidelines": "Store information accurately.",
#             "memory_format": "Use key-value pairs.",
#         }

#         _openai_sections = {"memory_guidelines": "<guidelines>Use XML tags.</guidelines>"}
#         _anthropic_sections = {"memory_guidelines": "## Guidelines\nUse markdown headers."}
#         _gemini_sections = {"memory_guidelines": "Guidelines: use plain text."}

#         # Provider-specific content block metadata (e.g. for prompt caching)
#         _anthropic_metadata = {"cache_control": {"type": "ephemeral"}}
#         _bedrock_metadata = {"cache_point": {"type": "default"}}

#     MemorySystemPrompt.get_content("openai", user_name="Alice")
#     MemorySystemPrompt.get_metadata("anthropic")  # {"cache_control": {"type": "ephemeral"}}
#     ```
#     """

#     _ROOT_PROMPT: ClassVar[str]
#     _sections: ClassVar[dict[str, str]]
#     _metadata: ClassVar[dict[str, Any]]

#     _root_template: ClassVar[Template]
#     _compiled_sections: ClassVar[dict[str, Template]]
#     _compiled_provider_sections: ClassVar[dict[str, dict[str, Template]]]

#     def __init_subclass__(cls, **kwargs: Any) -> None:
#         super().__init_subclass__(**kwargs)
#         if "_ROOT_PROMPT" not in cls.__dict__:
#             raise TypeError(f"{cls.__name__} must define _ROOT_PROMPT")
#         if "_sections" not in cls.__dict__:
#             cls._sections = {}
#         if "_metadata" not in cls.__dict__:
#             cls._metadata = {}
#         for provider in LLM_PROVIDERS:
#             for attr in (f"_{provider}_sections", f"_{provider}_metadata"):
#                 if attr not in cls.__dict__:
#                     setattr(cls, attr, {})

#         cls._root_template = Template(cls._ROOT_PROMPT, undefined=StrictUndefined)
#         cls._compiled_sections = {
#             name: Template(value, undefined=StrictUndefined) for name, value in cls._sections.items()
#         }
#         cls._compiled_provider_sections = {
#             provider: {
#                 name: Template(value, undefined=StrictUndefined)
#                 for name, value in getattr(cls, f"_{provider}_sections").items()
#             }
#             for provider in LLM_PROVIDERS
#         }

#     @classmethod
#     def get_content(cls, provider: LLMProvider, **kwargs: Unpack[_KwargsT]) -> str:
#         compiled_sections = {
#             **cls._compiled_sections,
#             **cls._compiled_provider_sections[provider],
#         }
#         rendered_sections = {name: section.render(**kwargs) for name, section in compiled_sections.items()}
#         leaves_sections = {**rendered_sections, **kwargs}
#         return cls._root_template.render(**leaves_sections)

#     @classmethod
#     def get_metadata(cls, provider: LLMProvider) -> dict[str, Any]:
#         return {**cls._metadata, **getattr(cls, f"_{provider}_metadata")}


class PromptFactory:
    _ALLOWED_TYPES_FOR_COMPONENTS = {str}

    """
    A lightweight class for constructing prompts tailored for generic LLMs.

    Key features:
    - Maintains modular prompt components, making them reusable across various prompt configurations.
    - Assembles prompts dynamically from an arbitrary combination of components, including strings, other PromptFactory instances, and additional data types.
    - Supports robust conditional logic, enabling the generation of context-specific prompts based on factors like the LLM provider or application use case.

    Two primary methods:
    - __init__: Initializes the PromptFactory with any number of core components (e.g., for use in an agent's long-term memory).
    - __call__: Generates the final prompt string, accepting an arbitrary set of attributes or context parameters at runtime (such as the model provider, user inputs, or application-specific data).

    ## Prompt components:
    The class must always have a _ROOT_PROMPT attribute, which is a formatting string that represents the final prompt.
    Then it can have an arbitrary number of additional class string attributes or methods returning strings, which represent the different components of the prompt.
        - Those can be defined as functions to enable conditional logic in the components too. The input those functions get will be the same as the input of the __call__ method in the invoking prompt factory class.
        - Every method that does not start with an underscore will be considered as a component and can be used in the final prompt.
    - Make sure that names given to the attributes are meaningful and descriptive, as those will be imported to other prompt factories as well. For instance, do not call it "guidelines", but rather "memory_guidelines" if the prompt factory is used for a memory agent.
    For instance:
    ```python
    class MemorySystemPrompt(PromptFactory):
        _ROOT_PROMPT = "{memory_role}\n{write_memory_instruction}\n{memory_guidelines}..."
        memory_role = "You are a helpful assistant. Your objective is to record important information about the user's requests and responses."
        
        def memory_guidelines(self, llm_provider: str) -> str:
            return {
                "openai": "<xxxx>STUFF</xxxx>",
                "anthropic": "#xxxx\nSTUFF",
            }[llm_provider]

        write_memory_instruction = "Write information in the memory using the following format: [timestamp] [user_request] [assistant_response]"
    ```

    ## Components aggregation automation:
    As the default behavior, the class supports some prompt components aggregation logic, to avoid boilerplate code and avoid errors in the final prompt construction. This is optional and can be overridden if more complex logic is needed.
    It works as follows. The __init__ method takes: 
    - a list of PromptFactory objects representing other prompt factories whose (some) components will be used to build the final prompt in self.
    - a dictionary of attributes containing additional components from arbitrary sources to be used in the final prompt as well, expressed as either strings or callable objects.

    When the __call__ method is called, the final prompt is constructed by formatting the _ROOT_PROMPT string with the components from the prompt_factories and the extra_attributes automatically based on the names' exact matches.

    If a component cannot be found in either the prompt_factories or the attributes dictionary, an error is raised.

    Note: It is a good idea to define tools/function-call names in a prompt factory as well. 
    """

    _ROOT_PROMPT:str = None

    @staticmethod
    def _is_format_string(string: str) -> bool:
        try:
            string.format()
            return False
        except:
            return True

    def __init__(self, prompt_factories:list=[], extra_attributes: dict={}):
        self.prompt_factories = prompt_factories
        self.extra_attributes = extra_attributes
        self.attributes = {}
        self._create_attribute_index()

    @staticmethod
    def _normalize_component(component):
        if isinstance(component, str):
            return lambda **kwargs: component
        elif callable(component):
            return component
        else:
            raise ValueError(f"Component {component} is not a string or a callable")

    def _check_for_component_name_conflict(self, component_name: str, source: str):
        if component_name in self.attributes:
            raise ValueError(f"Component {component_name} from {source} already exists")

    def _create_attribute_index(self):
        # Store all the components in a single dictionary for easy access, normalize all the components to be callable.
        for extra_attribute, value in self.extra_attributes.items():
            self._check_for_component_name_conflict(extra_attribute, "extra_attributes dictionary")
            self.attributes[extra_attribute] = self._normalize_component(value)

        # Add the components from the prompt factories and the current instance to the attributes dictionary.
        prompt_factories = self.prompt_factories + [self]
        for prompt_factory in prompt_factories:
            for attribute in prompt_factory.__dir__():
                if attribute.startswith('_'):
                    continue
                raw_component = getattr(prompt_factory, attribute)
                if not ( type(raw_component) in self._ALLOWED_TYPES_FOR_COMPONENTS or callable(raw_component) ):
                    continue
                self._check_for_component_name_conflict(attribute, f"prompt factory {prompt_factory.__class__.__name__}")
                self.attributes[attribute] = self._normalize_component(raw_component)


    def _compile_components(self, format_string: str, **kwargs):
        attributes = {attribute: component(**kwargs) for attribute, component in self.attributes.items()}
        while self._is_format_string(format_string):
            format_string = format_string.format(**attributes)
        return format_string

    def __call__(self, **kwargs) -> str:
        return self._compile_components(self._ROOT_PROMPT, **kwargs)


