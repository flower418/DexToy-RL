import argparse
import numpy as np

from dextoy.envs import AdroitButtonEnv
from dextoy.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env_button.yaml")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    env = AdroitButtonEnv(cfg["env"])

    obs, info = env.reset(seed=cfg["env"].get("seed", 0))

    print("obs shape:", obs.shape)
    print("action shape:", env.action_space.shape)
    print("initial info:", info)

    total_reward = 0.0

    for step in range(args.steps):
        # 先用零动作测试环境是否稳定。
        action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"step={step + 1:03d} "
            f"reward={reward:.3f} "
            f"depth={info['button_depth']:.4f} "
            f"raw_depth={info['raw_button_qpos']:.4f} "
            f"dist={info['min_fingertip_button_distance']:.4f} "
            f"contact={info['contact']} "
            f"success={info['success']}"
        )

        if terminated or truncated:
            break

    print("total reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()
