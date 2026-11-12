from llm.py import planner_model, prompt_model, sub_agents, review_model
from prompts.py import prompter, planner, sub_agents_prompt, reviewer, modifier
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Define the schema
class PlannerOutput(BaseModel):
    scene: int = Field(..., description="The scene number")
    plan: str = Field(..., description="The plan for the scene")

parser = PydanticOutputParser(pydantic_object=PlannerOutput)



def extract_code(prompt: str) -> str:
    """Extract code from a string that may contain markdown formatting."""
    if prompt.startswith("```") and prompt.endswith("```"):
        return "\n".join(prompt.split("\n")[1:-1])
    return prompt

plan = planner | planner_model | parser | RunnableLambda(lambda x: x.plan)
prompt = prompter | prompt_model | StrOutputParser()
sub_agents = sub_agents_prompt | sub_agents | StrOutputParser()| RunnableLambda(extract_code)
review = reviewer | review_model | StrOutputParser()
modify = modifier | sub_agents | StrOutputParser()| RunnableLambda(extract_code)


