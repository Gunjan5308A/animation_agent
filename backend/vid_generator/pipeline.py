from agents import plan, prompt, sub_agents, review, modify
import subprocess
from pathlib import Path

def generate_code(user_prompt: str) -> dict:
    """Generate a video plan from the user's prompt."""
    plan_pipeline = prompt | plan
    plan_output = plan_pipeline.invoke({"prompt": user_prompt})

    scene_codes = [sub_agents.invoke({"prompt": scene["plan"]}) for scene in plan_output]
    

    for i in range(len(scene_codes)):
        for j in range(5):
            scene_state = review.invoke({"prompt": scene_codes[i]})
            if scene_state == "approved":
                break
            scene_codes[i] = modify.invoke({"issue": scene_state,
                                             "prompt": scene_codes[i]})
            
    return scene_codes

