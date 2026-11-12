from agents import plan, prompt, sub_agents, review, modify
from merge import merge_videos
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

def generate_code(user_prompt: str) -> dict:
    """Generate a video plan from the user's prompt."""
    plan_pipeline = prompt | plan
    plan_output = plan_pipeline.invoke({"prompt": user_prompt})

    scene_codes = [sub_agents.invoke({"prompt": scene.plan}) for scene in plan_output]
    

    for i in range(len(scene_codes)):
        for j in range(2):
            scene_state = review.invoke({"prompt": scene_codes[i]})
            if scene_state == "approved":
                break
            scene_codes[i] = modify.invoke({"issue": scene_state,
                                             "prompt": scene_codes[i]})

            
    return scene_codes


def render_scenes(codes: list[str], output_dir: str = "rendered_scenes") -> list[str]:
    """Render each Manim scene code string into an mp4 and return their paths, in order."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rendered_files = []
    for i, code in enumerate(codes):
        match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
        if not match:
            raise ValueError(f"no Scene class found in scene {i}")
        scene_name = match.group(1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_file = Path(tmp_dir) / "scene.py"
            scene_file.write_text(code)

            subprocess.run(
                ["manim", "-qm", str(scene_file), scene_name, "--media_dir", tmp_dir],
                check=True,
            )

            rendered = next(Path(tmp_dir).rglob(f"{scene_name}.mp4"))
            dest = output_path / f"scene_{i}.mp4"
            shutil.copy(rendered, dest)
            rendered_files.append(str(dest))

    return rendered_files


def render_and_merge(codes: list[str], output: str = "final.mp4") -> str:
    """Render all scene codes, then merge them into a single final video."""
    scene_files = render_scenes(codes)
    return merge_videos(scene_files, output)

def generate_video(user_prompt: str, output: str = "final.mp4") -> str:
    """Generate a video from the user's prompt and return the path to the final mp4."""
    scene_codes = generate_code(user_prompt)
    final_video = render_and_merge(scene_codes, output)
    return final_video
