# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""AncnL 平地（Flat）任务配置。

继承 Rough 配置，做以下裁剪以适配平地训练：
1. 地形改为无限平面，去掉地形生成器与地形难度课程
2. 整体移除高度扫描传感器（scene.height_scanner）及 policy/critic 的
   height_scan 观测 —— 平地无需感知地形，模型为纯本体感知
3. base_height_l2 不再用地形传感器修正目标高度（sensor_cfg=None），
   直接以世界坐标系下的目标高度计算
4. 权重为 0 的奖励项直接置 None（省算力、日志更干净）

对应 gym 任务 id：RobotLab-Isaac-Velocity-Flat-Ancn-AncnL-v0
"""

from isaaclab.utils import configclass

from .rough_env_cfg import AncnLRoughEnvCfg


@configclass
class AncnLFlatEnvCfg(AncnLRoughEnvCfg):
    """AncnL 平地速度跟踪任务的完整环境配置。"""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        # 平地上 base_height_l2 直接用世界坐标目标高度，无需地形传感器修正
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        # 地形换成无限平面（无起伏、无边界），训练专注于速度跟踪本身
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        # 平地不需要激光/高度扫描：传感器与观测一起移除（纯本体感知）
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        # 没有多级地形，地形难度课程失去意义，直接关闭
        self.curriculum.terrain_levels = None

        # If the weight of rewards is 0, set rewards to None
        # 裁剪零权重奖励项（仅当直接实例化本类时执行，不影响 Rough 父类自身）
        if self.__class__.__name__ == "AncnLFlatEnvCfg":
            self.disable_zero_weight_rewards()
