import argparse
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="AdroitHandHammer-v1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", type=str, default="./media/videos/adroit_smoke_test.mp4") # 在项目根目录下运行
    args = parser.parse_args()

    gym.register_envs(gymnasium_robotics)

    env = gym.make(args.env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)

    print("env_id:", args.env_id)
    print("observation_type:", type(obs))
    print("observation_shape:", np.asarray(obs).shape)
    print("action space:", env.action_space)

    frames = []
    total_reward = 0

    for step in range(args.steps):
        action = env.action_space.sample() * 0.25
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        total_reward += float(reward)

        if terminated or truncated:
            break
    
    env.close()

    output_path = Path(args.video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames:
        imageio.mimsave(output_path, frames, fps=30)
        print("saved video:", output_path)
    else:
        print("no frame captured")

    print("episode return:", total_reward)
    print("episode length:", len(frames))
    print("final info:", info)

if __name__ == "__main__":
    main()