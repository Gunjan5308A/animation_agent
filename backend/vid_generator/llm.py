from langchain_openai import ChatOpenAI
from config import api_key, prompt_model, review_model, planner_model, sub_agents, base_url

def create_model(model_name: str):
    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        max_retries=2,
        timeout=None
    )

planner_model = create_model(planner_model)
prompt_model = create_model(prompt_model)
sub_agents = create_model(sub_agents)
review_model = create_model(review_model)