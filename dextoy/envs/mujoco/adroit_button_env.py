from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROBOT_JOINT_NAMES = (
    "ARTx", # arm translation
    "ARTy",
    "ARTz",
    "ARRx", # arm rotation
    "ARRy",
    "ARRz",
    "WRJ1", # wrist joints from the official Adroit hand
    "WRJ0",
    "FFJ3", # fore finger 食指，从指尖到掌心 3210
    "FFJ2",
    "FFJ1",
    "FFJ0",
    "MFJ3", # middle finger 中指
    "MFJ2",
    "MFJ1",
    "MFJ0",
    "RFJ3", # ring finger 无名指
    "RFJ2", 
    "RFJ1",
    "RFJ0",
    "LFJ4", # little finger 小拇指
    "LFJ3",
    "LFJ2",
    "LFJ1",
    "LFJ0",
    "THJ4", # thumb 大拇指
    "THJ3",
    "THJ2",
    "THJ1",
    "THJ0",
)

# site 表示虚拟标记点
TIP_SITE_NAMES = (
    "S_fftip", # 食指指尖
    "S_mftip",
    "S_rftip",
    "S_lftip",
    "S_thtip",
)

class AdroitButtonEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, config=None, render_mode=None):
        super().__init__()

        config = config or {}

        self.render_mode = render_mode
        self.max_episode_steps = int(config.get("max_episode_steps", 150))
        self.frame_skip = int(config.get("frame_skip", 5))

        model_xml = Path(config.get("model_xml", "dextoy/assets/mujoco/adroit_button.xml"))
        if not model_xml.is_absolute():
            model_xml = Path.cwd() / model_xml

        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)

        self.robot_joint_names = ROBOT_JOINT_NAMES
        self.tip_site_names = TIP_SITE_NAMES

        self.robot_qpos_addr = np.array(
            [self.model.jnt_qposadr[self._joint_id(name)] for name in self.robot_joint_names],
            dtype=np.int32,
        )
        self.robot_qvel_addr = np.array(
            [self.model.jnt_dofadr[self._joint_id(name)] for name in self.robot_joint_names],
            dtype=np.int32,
        )

        self.tip_site_ids = np.array(
            [self._site_id(name) for name in self.tip_site_names],
            dtype=np.int32,
        )

        self.button_site_id = self._site_id("S_button")
        self.button_joint_id = self._joint_id("button_slide")
        self.button_qpos_addr = int(self.model.jnt_qposadr[self.button_joint_id])

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        self.ctrl_mid = 0.5 * (self.ctrl_low + self.ctrl_high)
        self.ctrl_half_range = 0.5 * (self.ctrl_high - self.ctrl_low)

        self.success_depth = float(config.get("success_depth", 0.035))
        self.contact_radius = float(config.get("contact_radius", 0.07))

        reward_cfg = config.get("reward", {})
        self.distance_scale = float(reward_cfg.get("distance_scale", 2.0))
        self.contact_bonus = float(reward_cfg.get("contact_bonus", 0.25))
        self.press_scale = float(reward_cfg.get("press_scale", 40.0))
        self.success_bonus = float(reward_cfg.get("success_bonus", 20.0))
        self.action_smoothness_weight = float(reward_cfg.get("action_smoothness", 0.05))
        self.velocity_penalty_weight = float(reward_cfg.get("velocity_penalty", 0.002))

        self.step_count = 0
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.renderer = None

        obs_dim = (
            len(self.robot_joint_names) # 关节位置
            + len(self.robot_joint_names) # 关节速度
            + len(self.tip_site_names) * 3 # 指尖坐标 x,y,z
            + 3 # 按钮的 3 维坐标
            + 1 # 按钮按下的深度
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.prev_action[:] = 0.0

        mujoco.mj_resetData(self.model, self.data)

        # 使用官方 XML 的默认初始姿态。控制器初值设为当前关节位置，
        # 避免 reset 后 actuator 突然把手拉向 ctrlrange 中点。
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = np.clip(
            self.data.qpos[self.robot_qpos_addr],
            self.ctrl_low,
            self.ctrl_high,
        )

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        ctrl = self.ctrl_mid + action * self.ctrl_half_range
        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        info = self._get_info()
        reward = self._compute_reward(action, info)

        terminated = bool(info["success"])
        truncated = self.step_count >= self.max_episode_steps

        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != "rgb_array":
            return None

        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        self.renderer.update_scene(self.data, camera="front")
        return self.renderer.render()
    
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _get_obs(self):
        joint_pos = self.data.qpos[self.robot_qpos_addr]
        joint_vel = self.data.qvel[self.robot_qvel_addr]
        fingertip_pos = self.data.site_xpos[self.tip_site_ids].reshape(-1)
        button_pos = self.data.site_xpos[self.button_site_id]
        button_depth = self._button_depth()

        obs = np.concatenate(
            [
                joint_pos,
                joint_vel,
                fingertip_pos,
                button_pos,
                np.array([button_depth], dtype=np.float64),
            ]
        )

        return obs.astype(np.float32)
    
    def _get_info(self):
        fingertip_pos = self.data.site_xpos[self.tip_site_ids]
        button_pos = self.data.site_xpos[self.button_site_id]

        distances = np.linalg.norm(fingertip_pos - button_pos[None, :], axis=1)
        min_distance = float(np.min(distances))

        button_depth = self._button_depth()
        contact = min_distance <= self.contact_radius
        success = button_depth >= self.success_depth

        return {
            "success": bool(success),
            "contact": bool(contact),
            "button_depth": float(button_depth),
            # 调试用：MuJoCo 里 button_slide 的原始 qpos。
            # 如果 raw 是负数但 button_depth 是 0，说明按钮运动方向或限位需要调整。
            "raw_button_qpos": float(self.data.qpos[self.button_qpos_addr]),
            "min_fingertip_button_distance": min_distance,
            "episode_step": self.step_count,
        }
    
    def _compute_reward(self, action, info):
        min_distance = info["min_fingertip_button_distance"]
        button_depth = info["button_depth"]

        reward = 0.0
        reward += -self.distance_scale * min_distance
        reward += self.contact_bonus if info["contact"] else 0.0
        reward += self.press_scale * button_depth
        reward += self.success_bonus if info["success"] else 0.0

        action_delta = action - self.prev_action
        reward -= self.action_smoothness_weight * float(np.mean(action_delta ** 2))
        reward -= self.velocity_penalty_weight * float(np.linalg.norm(self.data.qvel))

        return float(reward)

    def _button_depth(self):
        # MuJoCo constraints can have tiny numerical violation around the lower
        # joint limit. For the task definition, negative press depth has no
        # meaning, so expose a non-negative value to reward/info/obs.
        return max(0.0, float(self.data.qpos[self.button_qpos_addr]))
    
    def _joint_id(self, name):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"Missing joint: {name}")
        return joint_id
    
    def _site_id(self, name):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise KeyError(f"Missing site: {name}")
        return site_id
    
