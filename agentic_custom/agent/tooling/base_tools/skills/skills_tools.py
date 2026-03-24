from ... import ToolsContext, Tool, Argument
from ...tools_context import tool
from . import SkillsManager, DEFAULT_SKILL_FUNCTION_DESCRIPTION, SKILLS_TOOL_NAME


class SkillsToolsContext(ToolsContext):
    def __init__(self, skills_directory: str = None, skill_function_description: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.skills_manager = SkillsManager(skills_directory)

        if skill_function_description is None:
            skill_function_description = DEFAULT_SKILL_FUNCTION_DESCRIPTION
        self.skill_function_description = skill_function_description

    def get_skills_list(self) -> str:
        return self.skills_manager.list_skills()

    @tool
    def skill_tool_definition(self):
        def f(skill: str, args: str = None):
            skill_content = self.skills_manager.get_skill(skill, args)
            if skill_content is None:
                return f"Error: Skill '{skill}' not found"
            return skill_content
        return Tool(
            name=SKILLS_TOOL_NAME,
            function=...,
            description=self.skill_function_description,
            arguments=[
                Argument(
                    name="skill",
                    description="""The skill name. E.g., "commit", "review-pr", or "pdf" """,
                    type="string",
                ),
                Argument(
                    name="args",
                    description='Optional arguments for the skill',
                    type="string",
                    required=False,
                )
            ]
        )
