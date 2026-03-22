import os, glob
import yaml
from .....config import get_output_directory

SKILLS_DIRECTORY_NAME = "skills"
SKILLS_TOOL_NAME = "skill"

DEFAULT_SKILL_FUNCTION_DESCRIPTION = """Execute a skill within the main conversation

When users ask you to perform tasks, check if any of the available skills match. Skills provide specialized capabilities and domain knowledge.

When users reference a "slash command" or "/<something>" (e.g., "/commit", "/review-pr"), they are referring to a skill. Use this tool to invoke it.

How to invoke:

- Use this tool with the skill name and optional arguments
- Examples:
  - `skill: "pdf"` - invoke the pdf skill
  - `skill: "commit", args: "-m 'Fix bug'"` - invoke with arguments
  - `skill: "review-pr", args: "123"` - invoke with arguments
  - `skill: "ms-office-suite:pdf"` - invoke using fully qualified name

Important:

- When a skill matches the user's request, this is a BLOCKING REQUIREMENT: invoke the relevant Skill tool BEFORE generating any other response about the task
- NEVER mention a skill without actually calling this tool
- Do not invoke a skill that is already running
- Do not use this tool for built-in CLI commands (like /help, /clear, etc.)
- If you see a <command-name> tag in the current conversation turn, the skill has ALREADY been loaded - follow the instructions directly instead of calling this tool again"""


SKIP_DIRECTORIES = {".git", "node_modules", "__pycache__", ".venv"}
MAX_RESOURCE_FILES = 200


class Skill:
    def __init__(self, path: str):
        self.path = path
        self.base_directory = os.path.dirname(path)
        self.name, self.description, self.content = self._parse_frontmatter(self.path)
        self.resources = self._discover_resources()

    @staticmethod
    def _parse_frontmatter(path: str) -> tuple[str, str, str]:
        """Parse a SKILL.md file returning (name, description, body content).

        The file must have YAML frontmatter delimited by '---' lines
        containing at least 'name' and 'description' fields, followed
        by the body content.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()

        if not raw.startswith("---"):
            raise ValueError(f"Skill file {path} missing YAML frontmatter")

        end_index = raw.index("---", 3)
        frontmatter_str = raw[3:end_index]
        content = raw[end_index + 3:].strip()

        frontmatter = yaml.safe_load(frontmatter_str)
        if not isinstance(frontmatter, dict):
            raise ValueError(f"Skill file {path} has invalid YAML frontmatter")

        name = frontmatter.get("name")
        description = frontmatter.get("description")
        if not name or not description:
            raise ValueError(f"Skill file {path} missing required 'name' or 'description' in frontmatter")

        return name, description, content

    def _discover_resources(self) -> list[str]:
        """Enumerate bundled resource files relative to the skill directory.

        Walks the skill's base directory, skipping hidden dirs and common
        non-content directories. Returns paths relative to base_directory.
        Files are listed but not read (tier-3 progressive disclosure).
        """
        resources = []
        for dirpath, dirnames, filenames in os.walk(self.base_directory):
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRECTORIES and not d.startswith(".")
            ]
            for filename in filenames:
                if filename == "SKILL.md":
                    continue
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, self.base_directory)
                resources.append(rel_path)
                if len(resources) >= MAX_RESOURCE_FILES:
                    return resources
        return resources


class SkillsManager:
    def __init__(self, skills_directory: str = None):
        if skills_directory is None:
            self.skills_directory = os.path.join(get_output_directory(), SKILLS_DIRECTORY_NAME)
        else:
            self.skills_directory = skills_directory
        os.makedirs(self.skills_directory, exist_ok=True)

        self.skills_paths = glob.glob(os.path.join(self.skills_directory, "**/SKILL.md"))
        print(f'Found {len(self.skills_paths)} skills in {self.skills_directory}')

        self.loaded_skills = {}
        self.load_skills()

    def load_skills(self):
        for skill_path in self.skills_paths:
            skill = Skill(skill_path)
            self.loaded_skills[skill.name] = skill

    def list_skills(self) -> str:
        if not self.loaded_skills:
            return "<available_skills>\n</available_skills>"
        lines = ["<available_skills>"]
        for skill in self.loaded_skills.values():
            lines.append("  <skill>")
            lines.append(f"    <name>{skill.name}</name>")
            lines.append(f"    <description>{skill.description}</description>")
            lines.append(f"    <location>{skill.path}</location>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
        return "\n".join(lines)

    def get_skill(self, skill_name: str) -> str | None:
        skill = self.loaded_skills.get(skill_name)
        if skill is None:
            return None

        lines = [f'<skill_content name="{skill.name}">']
        lines.append(skill.content)
        lines.append("")
        lines.append(f"Skill directory: {skill.base_directory}")
        lines.append("Relative paths in this skill are relative to the skill directory.")
        if skill.resources:
            lines.append("")
            lines.append("<skill_resources>")
            for resource in skill.resources:
                lines.append(f"  <file>{resource}</file>")
            lines.append("</skill_resources>")
        lines.append("</skill_content>")
        return "\n".join(lines)
