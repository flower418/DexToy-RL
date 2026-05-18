from pathlib import Path
import imageio.v2 as imageio


def save_video(frames, path, fps=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(frames) == 0:
        raise ValueError("No frames to save.")

    imageio.mimsave(path, frames, fps=fps)