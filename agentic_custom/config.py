"""Process-wide configuration shared across agentic_custom."""

import os
from pathlib import Path
from typing import Union

# Default root for library-generated files (under the user's home directory).
DEFAULT_OUTPUT_DIRECTORY: Path = Path.home() / ".agentic_custom"

_OUTPUT_DIRECTORY_ENV_VAR = "AGENTIC_CUSTOM_OUTPUT_DIRECTORY"


class AgenticConfig:
    """
    Library-wide settings.

    Use the shared instance from :func:`get_config` or ``config``. Add new
    fields here as the library grows; read them at use time rather than caching
    copies in other modules.

    On startup, ``output_directory`` is taken from the environment variable
    ``AGENTIC_CUSTOM_OUTPUT_DIRECTORY`` when set and non-empty; otherwise
    :data:`DEFAULT_OUTPUT_DIRECTORY` is used.
    """

    def __init__(self) -> None:
        env_value = os.environ.get(_OUTPUT_DIRECTORY_ENV_VAR)
        if env_value is not None and env_value.strip() != "":
            self._output_directory = Path(env_value).expanduser().resolve(
                strict=False
            )
        else:
            self._output_directory = Path(DEFAULT_OUTPUT_DIRECTORY)

    @property
    def output_directory(self) -> Path:
        return self._output_directory

    @output_directory.setter
    def output_directory(self, value: Union[str, Path]) -> None:
        self._output_directory = Path(value).expanduser().resolve(strict=False)


config = AgenticConfig()


def get_config() -> AgenticConfig:
    return config


def set_output_directory(path: Union[str, Path]) -> None:
    config.output_directory = path


def get_output_directory() -> Path:
    return config.output_directory


__all__ = [
    "AgenticConfig",
    "DEFAULT_OUTPUT_DIRECTORY",
    "config",
    "get_config",
    "get_output_directory",
    "set_output_directory",
]
