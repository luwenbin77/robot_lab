# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""AncnL 四轮腿机器人的 Rough 环境配置。

在通用模板 velocity_env_cfg.py 基础上定义 ancnL 的专属差异：
- 动作：12 个腿关节用位置控制，4 个轮子用速度控制
- 观测：纯本体感知（关节/机身角速度/重力投影/速度命令），无线速度、无高度扫描
- 奖励：以速度命令跟踪为主目标，辅以能耗、姿态、接触等正则化惩罚
- 域随机化：出生位姿、质量、质心、外力、执行器增益随机化

Flat 任务（flat_env_cfg.py）继承本配置，仅替换地形并关闭高度扫描。
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.ancnL import ANCNL_CFG  # isort: skip


@configclass
class AncnLActionsCfg(ActionsCfg):
    """Action specifications for the MDP.

    混合动作空间（joint_names 留空作占位，__post_init__ 中按腿/轮分别绑定）：
    - joint_pos：腿关节位置增量（use_default_offset=True，相对默认站立姿态偏移）
    - joint_vel：轮关节目标角速度
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=20.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class AncnLRewardsCfg(RewardsCfg):
    """Reward terms for the MDP.

    在基类 RewardsCfg 基础上，为 4 个轮关节补充独立的 L2 惩罚项
    （速度/加速度/扭矩），权重默认 0，__post_init__ 中按需开启。
    """

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class AncnLRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """AncnL Rough 环境主配置。"""

    actions: AncnLActionsCfg = AncnLActionsCfg()
    rewards: AncnLRewardsCfg = AncnLRewardsCfg()

    # 常用刚体/关节名，供各配置段引用
    base_link_name = "base_link"  # 机身刚体（高度/姿态惩罚与外力随机化的目标）
    foot_link_name = ".*_wheel"  # 本机器人的"足端"即 4 个轮子（接触类奖励围绕它）

    # fmt: off
    # 12 个腿关节：fl/fr/hl/hr = 左前/右前/左后/右后
    leg_joint_names = [
        "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
        "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
        "hl_hip_joint", "hl_thigh_joint", "hl_calf_joint",
        "hr_hip_joint", "hr_thigh_joint", "hr_calf_joint",
    ]
    # 4 个驱动轮关节
    wheel_joint_names = [
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    ]
    # 全部 16 个执行关节（腿 + 轮）
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # 用 ancnL 专属资产替换模板中的默认机器人
        self.scene.robot = ANCNL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 高度扫描传感器挂载到机身（Rough 地形测地表高度用；Flat 任务会整体移除）
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # 关节位置观测剔除 4 个轮关节：轮子连续旋转，其"角度"没有物理意义
        # （policy 与 critic 保持一致）
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        # 各观测项缩放：把不同量纲调到相近数量级，利于网络训练
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 纯本体感知：机身线速度与高度扫描均不进入 policy 观测
        # （线速度在真实机器人上难以直接测量，去掉以保持 sim-to-real 一致）
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        # 观测覆盖全部 16 个关节（12 腿 + 4 轮），再由 func 内部剔除轮关节位置
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # 混合动作：腿 = 位置增量（相对默认姿态），轮 = 目标角速度
        # reduce action scale
        # hip 动作幅度 0.125、thigh/calf 0.25：限制单步腿部动作量，保持稳定
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # 轮速动作缩放：10.0 rad/s × 轮半径 0.11 m = 1.1 m/s 极速。
        # 旧值 5.0 时极速仅 0.55 m/s，追不上 ±1.0 m/s 的速度命令（实测瓶颈）
        self.actions.joint_vel.scale = 10.0
        # clip 放开到 ±100（实际由 scale 与执行器限幅兜底）
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        # 两组动作分别绑定腿关节与轮关节
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # 出生随机化：水平 ±0.5 m、高度 +0~0.2 m、roll/pitch/yaw 全随机 ——
        # 机器人会以任意翻滚姿态出生，需要自学翻身站起
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-3.14, 3.14),
                "pitch": (-3.14, 3.14),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        # 质量随机化：机身与非机身刚体分开设置
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # 质心随机化与外力扰动均作用在机身
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        # 终止惩罚关闭：不做摔机重罚，靠正奖励引导（time_out 正常结束无惩罚）
        self.rewards.is_terminated.weight = 0

        # Root penalties（机身惩罚）
        # 竖直方向速度 L2：抑制上下蹦跳
        self.rewards.lin_vel_z_l2.weight = -2.0
        # 机身角速度 L2：抑制侧倾/俯仰晃动
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 机身水平惩罚：关闭（由 upward 奖励兜底保持机身朝上）
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.50
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties（关节正则：小系数能耗惩罚只作用于腿，轮子单列见下）
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        # 关节限位惩罚：重罚顶到软限位（防止关节卡死）
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        # 无命令时腿必须保持默认姿态（防原地乱动）
        self.rewards.stand_still.weight = -2.0
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        # 腿偏离默认关节角的惩罚：轮腿机器人靠轮子滚动移动，腿应保持站姿
        # （无命令时该项内部再乘 stand_still_scale=5 加倍抑制）
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = self.foot_link_name
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 对角腿镜像同步（fl↔hr、fr↔hl）：鼓励规整、对称的腿姿态
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["fl_(hip|thigh|calf).*", "hr_(hip|thigh|calf).*"],
            ["fr_(hip|thigh|calf).*", "hl_(hip|thigh|calf).*"],
        ]

        # Action penalties
        # 相邻两步动作差分 L2：动作平滑
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor（接触惩罚）
        # 除轮子外的任何刚体接触都重罚（防摔、防底盘磕地）
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # 轮子接地冲击力限幅
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards（训练主目标）
        # 线速度跟踪：exp(-|v_cmd-v|²/std²) 指数核，误差越小奖励越接近满值 3.0
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # 角速度（转向）跟踪
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        # 以下迈步/步态类奖励（air_time/contact/stumble/slide/height/gait）
        # 对轮腿机器人意义不大：移动由轮子滚动完成，腿不需要摆动节律，全部关闭
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 无命令时轮子着地给少量奖励（鼓励保持四轮站立不动）
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.1
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.42
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("fl_wheel", "hr_wheel"), ("fr_wheel", "hl_wheel"))
        # 机身朝上奖励：防摔倒的主要正奖励
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        # 裁剪零权重奖励项：仅当直接实例化 Rough 配置时执行（Flat 子类走自己的判断）
        if self.__class__.__name__ == "AncnLRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # 轮腿机器人的"脚"就是轮子，无法定义非法接触，关闭该终止条件
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # 命令难度课程关闭：命令范围恒定为 ±1.0，不做渐进放大
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # 沿用基类默认命令范围：lin_vel_x/y ∈ ±1.0 m/s，ang_vel_z ∈ ±1.0 rad/s
        # （若需收紧到轮子实际能力范围内，在此覆写即可）
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.5, 1.5)
