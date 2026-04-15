#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    act=None,
)


# SampleData用于在aisrv和learner之间传递训练样本
# 必须使用整数定义维度（不能用None），框架层会自动生成FIELD_DIMS并处理序列化
SampleData = create_cls(
    "SampleData",
    obs=79,  # 观测维度，对应Config.FEATURE_VECTOR_SHAPE[0]
    legal_actions=8,  # 合法动作维度
    actions=1,  # 动作维度（标量）
    probs=8,  # 动作概率分布维度
    rewards=1,  # 奖励（标量）
    advantages=1,  # 优势函数（标量）
    values=1,  # 价值函数（标量）
    dones=1,  # 是否结束（标量）
    next_values=1,
    reward_sum=1,
    # 根据你的实际算法需求添加其他字段
)


def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    reward = 0.0
    survive_reward = 0.01
    return [survive_reward + reward]


def sample_process(list_game_data):
    for i in range(len(list_game_data) - 1):
        list_game_data[i].next_values = list_game_data[i + 1].values

    gae = 0.0
    gamma = Config.GAMMA
    lamda = 0.95
    for sample in reversed(list_game_data):
        delta = -sample.values + sample.rewards + gamma * sample.next_values
        gae = gae * gamma * lamda + delta
        sample.advantages = gae
        sample.reward_sum = gae + sample.values
    return list_game_data
