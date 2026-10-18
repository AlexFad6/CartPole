import pybullet as p
import pybullet_data
import gymnasium as gym
import numpy as np

class ParametricInvertedPendulum(gym.Env):
    """
    control_mode: "acceleration" → 动作输出为各关节力矩（对应加速度控制）
                  "velocity"     → 动作输出为各关节目标速度（速度伺服）
    """
    def __init__(self, num_segments=1, segment_lengths=None,
                 gravity=9.81, control_mode="acceleration",
                 render_mode=None):
        super().__init__()
        self.num_segments = num_segments
        self.segment_lengths = segment_lengths or [0.5] * num_segments
        assert len(self.segment_lengths) == num_segments
        self.gravity = gravity
        self.control_mode = control_mode
        self.render_mode = render_mode

        # 动作空间：1 个自由度对应一个关节（简化：只控制基座关节或每节关节）
        # 这里假设控制最底部的驱动关节，若需控制每个关节可扩展为 Box(-limit, limit, (num_segments,))
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 观察空间：各关节角度 + 角速度，以及基座位置/速度
        obs_dim = 2 * self.num_segments + 2
        high = np.ones(obs_dim, dtype=np.float32) * 10.0
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.physics_client = None
        self.pendulum_ids = []
        self.joint_ids = []
        self._build_world()

    def _build_world(self):
        """在 PyBullet 中动态创建多节倒立摆"""
        self.physics_client = p.connect(p.DIRECT if self.render_mode is None else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.gravity)  # ← 重力参数直接生效

        # 地面/基座
        self.base_id = p.loadURDF("plane.urdf")

        # 逐节创建连杆
        parent = self.base_id
        parent_frame_pos = [0, 0, 0]
        for i, length in enumerate(self.segment_lengths):
            # 每节简化为细长圆柱，长度 = segment_lengths[i]
            col_id = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.02,
                height=length,
            )
            vis_id = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.02,
                length=length,
                rgbaColor=[0.8, 0.2, 0.2, 1.0] if i == 0 else [0.2, 0.4, 0.8, 1.0],
            )
            # 关节连接位置：每节的中点相对于父连杆的偏移
            joint_pos = [0, 0, length / 2 + (0 if i == 0 else self.segment_lengths[i-1] / 2)]
            link_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[0, 0, joint_pos[2]],
            )
            # 创建铰链关节（绕 Y 轴旋转，模拟倒立摆）
            joint_id = p.createConstraint(
                parentBodyUniqueId=parent,
                parentLinkIndex=-1,
                childBodyUniqueId=link_id,
                childLinkIndex=-1,
                jointType=p.JOINT_REVOLUTE,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, joint_pos[2] - (0 if i == 0 else self.segment_lengths[i-1] / 2)],
                childFramePosition=[0, 0, -length / 2],
            )
            self.pendulum_ids.append(link_id)
            self.joint_ids.append(joint_id)
            parent = link_id

    def _get_obs(self):
        obs = []
        for link_id in self.pendulum_ids:
            pos, orn = p.getBasePositionAndOrientation(link_id)
            # 将四元数转为欧拉角，取绕 Y 轴的倾角
            euler = p.getEulerFromQuaternion(orn)
            obs.append(euler[1])  # 角度
            vel, ang_vel = p.getBaseVelocity(link_id)
            obs.append(ang_vel[1])  # 角速度
        # 补充基座状态（简化）
        obs.extend([0.0, 0.0])  # 基座位置/速度占位，按需修改
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # 仅控制最底部关节（可根据需要扩展为多关节控制）
        joint_id = self.joint_ids[0]
        if self.control_mode == "acceleration":
            # 力矩模式：action 映射为力矩
            torque = action[0] * 5.0  # 力矩范围 ±5 N·m，可调
            p.setJointMotorControl2(
                bodyUniqueId=self.pendulum_ids[0],
                jointIndex=0,
                controlMode=p.TORQUE_CONTROL,
                force=torque,
            )
        elif self.control_mode == "velocity":
            # 速度模式：action 映射为目标角速度
            target_vel = action[0] * 5.0  # rad/s
            p.setJointMotorControl2(
                bodyUniqueId=self.pendulum_ids[0],
                jointIndex=0,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_vel,
                force=10.0,  # 速度伺服的最大力
            )
        else:
            raise ValueError("control_mode 必须为 'acceleration' 或 'velocity'")

        p.stepSimulation()
        obs = self._get_obs()

        # 奖励：保持直立（角度接近 0）+ 惩罚角速度过大
        angles = obs[0::2][:self.num_segments]
        ang_vels = obs[1::2][:self.num_segments]
        reward = -np.sum(np.square(angles)) - 0.01 * np.sum(np.square(ang_vels))

        # 终止条件：任一节倾角超过 90°
        terminated = bool(np.any(np.abs(angles) > np.pi / 2))
        truncated = False  # 可配合 TimeLimit wrapper
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -self.gravity)
        self._build_world()
        return self._get_obs(), {}