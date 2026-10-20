# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for AncnL wheeled-legged robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

# AncnL 轮足机器人的资产配置（ArticulationCfg）。
# 作用：把 URDF 模型接入 IsaacLab，供强化学习环境（velocity 跟踪任务）实例化使用。
# 调用链：本文件被 tasks/.../wheeled/ancnL/rough_env_cfg.py 导入，
#         通过 ANCNL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") 注入 scene.robot，
#         并在 ancnL/__init__.py 中注册为 RobotLab-Isaac-Velocity-Flat/Rough-Ancn-AncnL-v0。
ANCNL_CFG = ArticulationCfg(
    # ---------- 生成（spawn）配置：运行时把 URDF 转换为 USD ----------
    spawn=sim_utils.UrdfFileCfg(
        # 自由浮动基座，可移动，适用于行走/平衡类 RL
        fix_base=False,
        # 合并固定关节，减少刚体数量与仿真开销
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        # 模型路径：ISAACLAB_ASSETS_DATA_DIR = source/robot_lab/data
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ancnL/ancnL_description/urdf/ancnL.urdf",
        # 开启接触力传感器（接触检测、足端受力/接触类奖励项依赖它）
        activate_contact_sensors=True,
        # ---------- 刚体属性：RL 训练的宽松设置 ----------
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 受重力，机器人从初始高度落下
            retain_accelerations=False,
            linear_damping=0.0,  # 不额外加阻尼，避免干扰动力学学习
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,  # 限制穿模恢复速度，防止数值爆炸
        ),
        # ---------- Articulation 根属性 ----------
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 关闭自碰撞：加速仿真，代价是腿间互穿不会被物理阻挡
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        # URDF 转换阶段的默认关节驱动增益置零，
        # 实际增益完全由下面的 actuators 覆盖，避免两处配置冲突
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    # ---------- 初始状态 ----------
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),  # 出生高度 0.5 m
        joint_pos={
            # 站立姿态：髋关节归零、大腿前抬、小腿后折、轮子归零
            ".*_hip_joint": 0.0,
            ".*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_wheel_joint": 0.0,
        },
        joint_vel={".*": 0.0},  # 所有关节初速为 0
    ),
    # 软关节限位取硬限位的 90%，防止策略把关节顶到极限
    soft_joint_pos_limit_factor=0.9,
    # ---------- 执行器分组：轮足机器人的关键设计 ----------
    actuators={
        # 腿部：除轮子外的全部关节（hip/thigh/calf），带刚度的位置伺服
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["^(?!.*_wheel_joint).*"],
            effort_limit_sim=113.0,  # 力矩上限 N·m
            velocity_limit_sim=15.708,  # 关节速度上限 rad/s
            stiffness=25.0,  # 有刚度 => 位置跟踪
            damping=0.5,
            friction=0.0,
        ),
        # 轮子：stiffness=0，只保留阻尼，等效为速度/力矩驱动的连续转动
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit_sim=19.3,
            velocity_limit_sim=94.248,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of AncnL wheeled-legged robot.

URDF 共 16 个关节：fl/fr/hl/hr × {hip, thigh, calf, wheel}_joint，
其中 *_wheel_joint 为 continuous 型，可无限旋转，与上面两个执行器组的正则匹配一一对应。

注意：配置依赖关节命名。若 URDF 重命名关节（如 hip -> hip_roll）或改动层级，
需同步更新 joint_pos、actuators.joint_names_expr，以及环境配置中的
base_link_name 与 foot_link_name。
"""
