from llm import planner_model, prompt_model, sub_agents as sub_agent_model, review_model
from prompts import prompter, planner, sub_agents_prompt, reviewer, modifier
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


class ScenePlan(BaseModel):
    scene: int = Field(..., description="The scene number")
    plan: str = Field(..., description="The plan for the scene")


class PlannerOutput(BaseModel):
    scenes: list[ScenePlan] = Field(..., description="Ordered list of scene plans")


parser = PydanticOutputParser(pydantic_object=PlannerOutput)

REVIEW_FORMAT_INSTRUCTIONS = (
    "Respond with exactly the single word `approved` if no change is needed. "
    "Otherwise, respond with a concise, actionable description of the issue to fix."
)


def extract_code(prompt: str) -> str:
    """Extract code from a string that may contain markdown formatting."""
    if prompt.startswith("```") and prompt.endswith("```"):
        return "\n".join(prompt.split("\n")[1:-1])
    return prompt


def with_defaults(**defaults):
    """Fill in any keys missing from the invoked dict with the given defaults."""
    return RunnableLambda(lambda data: {**defaults, **data})


prompt = prompter | prompt_model | StrOutputParser()

plan = (
    RunnableLambda(lambda brief: {"prompt": brief, "format_instructions": parser.get_format_instructions()})
    | planner
    | planner_model
    | parser
    | RunnableLambda(lambda output: output.scenes)
)

sub_agents = (
    with_defaults(docs="", global_doc="")
    | RunnableLambda(lambda data: {**data, "section_doc": data["prompt"]})
    | sub_agents_prompt
    | sub_agent_model
    | StrOutputParser()
    | RunnableLambda(extract_code)
)

review = (
    with_defaults(format_instructions=REVIEW_FORMAT_INSTRUCTIONS)
    | reviewer
    | review_model
    | StrOutputParser()
)

modify = (
    with_defaults(docs="")
    | modifier
    | sub_agent_model
    | StrOutputParser()
    | RunnableLambda(extract_code)
)
