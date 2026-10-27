# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for AncnL wheeled-legged robot.

AncnL 四轮腿机器人（wheel-legged）资产配置。整机 16 个执行关节：
- 4 条腿 × (hip 侧摆 + thigh 大腿 + calf 小腿) = 12 个腿部关节，位置式 PD 控制
- 4 个驱动轮：速度控制（无位置刚度、纯阻尼），由 action 直接给目标角速度
URDF 零位（全 0 关节角）即标准站立姿态：物理仿真实测机身稳定在约 0.50 m、
四轮均匀着地、机身水平；初始 z 取 0.54 略高于静稳高度，落下即可站稳。
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


ANCNL_CFG = ArticulationCfg(
    # ---- 生成（spawn）设置：从 URDF 导入并配置物理属性 ----
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,  # 机身不固定（自由浮动，可以摔倒）
        merge_fixed_joints=True,  # 合并固定关节，减少自由度、加速仿真
        replace_cylinders_with_capsules=False,  # 保留原始圆柱碰撞体（轮子必须用圆柱）
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/ancnL/ancnL_description/urdf/ancnL.urdf",
        activate_contact_sensors=True,  # 开启接触传感器（接触类奖励/终止需要）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 受重力
            retain_accelerations=False,
            linear_damping=0.0,  # 不加人为阻尼，动力学全部交给执行器
            angular_damping=0.0,
            max_linear_velocity=1000.0,  # 解除刚体速度上限（默认值对高速轮子不够）
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,  # 限制穿透修复速度，避免接触弹射
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 关闭自碰撞（腿与机身间隙小，开启易误触发）
            solver_position_iteration_count=4,  # 位置求解迭代数（影响接触稳定性）
            solver_velocity_iteration_count=1,
        ),
        # URDF 导入时的关节驱动增益清零：驱动完全由下方 actuators 接管，
        # 避免 URDF 自带 PD 与 Isaac Lab 执行器模型叠加干扰
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    # ---- 初始状态：URDF 零位即站立姿态（实测机身 ~0.50 m，四轮齐平、机身水平）----
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.54),  # 初始高度略高于静稳高度 ~0.50，下落少许即可站稳
        joint_pos={
            ".*_hip_joint": 0.0,  # 髋侧摆归零
            ".*_thigh_joint": 0.0,  # 大腿归零（前后腿对称站立）
            ".*_calf_joint": 0.0,  # 小腿归零
            ".*_wheel_joint": 0.0,  # 轮子角度无物理意义，归零即可
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,  # 软限位包络系数（关节限位惩罚基于软限位计算）
    # ---- 执行器：腿部位置式直流电机，轮子速度控制 ----
    actuators={
        # 腿部：PD 位置控制。stiffness=80 是撑住机身的关键 ——
        # 之前用 25 时，叠加 reset 时刻的增益随机化（×0.5~2.0）最差仅 12.5，
        # 撑不住机身、训练中会直接趴地
        "leg": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=113.0,  # 扭矩上限（与真实电机一致）
            saturation_effort=113.0,  # 力矩饱和点（超过后按 DC 电机转速-扭矩曲线衰减）
            velocity_limit=15.708,  # 腿关节转速上限（rad/s）
            stiffness=80.0,  # P 增益
            damping=2.0,  # D 增益
            friction=0.0,
        ),
        # 轮子：速度控制（stiffness=0，无位置回复力），阻尼提供速度跟踪阻尼项
        "wheel": DCMotorCfg(
            joint_names_expr=[".*_wheel_joint"],
            effort_limit=19.3,
            saturation_effort=19.3,
            velocity_limit=94.248,  # 理论极速 ~10 m/s（轮半径 0.11 m），远超训练需求
            stiffness=0.0,  # 无位置刚度：轮子只受速度指令驱动，不回位
            damping=0.6,  # 阻尼项
            friction=0.0,
        ),
    },
)
"""Configuration of AncnL wheeled-legged robot.
"""
