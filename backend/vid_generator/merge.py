import subprocess
import tempfile
from pathlib import Path


def merge_videos(scene_files: list[str], output: str = "final.mp4") -> str:
    """Merge scene mp4s (in given order) into one video using ffmpeg concat."""
    if not scene_files:
        raise ValueError("no scene files to merge")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for path in scene_files:
            f.write(f"file '{Path(path).resolve()}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # scenes rendered with different codecs/params -> re-encode
        cmd[cmd.index("-c"):] = ["-c:v", "libx264", "-c:a", "aac", output]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    return output
