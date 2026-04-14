#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共59维）
    FEATURES = [
        5,      # 英雄自身特征（增加了加速Buff状态）
        5,      # 怪物1特征
        5,      # 怪物2特征
        2,      # 宝箱1特征
        2,      # 宝箱2特征
        2,      # 宝箱3特征
        2,      # 怪物1移动向量
        2,      # 怪物2移动向量
        16,     # 地图特征
        16,     # 合法动作（增加了8个闪现方向）
        2,      # 进度特征
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向 + 8个闪现方向 = 16维
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.01  # 调高熵系数以鼓励探索
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
