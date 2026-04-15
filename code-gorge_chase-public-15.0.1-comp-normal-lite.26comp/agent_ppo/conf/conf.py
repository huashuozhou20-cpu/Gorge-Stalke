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

    # Feature dimensions / 特征维度（共89维）
    FEATURES = [
        4,      # 英雄自身特征
        5,      # 怪物1特征
        5,      # 怪物2特征
        4,      # 怪物1威胁向量
        4,      # 怪物2威胁向量
        2,      # 宝箱1特征
        2,      # 宝箱2特征
        2,      # 宝箱3特征
        2,      # 最近宝箱引导特征
        49,     # 局部地图特征（7x7多值掩码）
        8,      # 合法动作掩码
        2,      # 进度特征
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：8个移动方向 + 8个闪现方向 = 16维
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # Network architecture / 网络结构
    HIDDEN_DIM1 = 128
    HIDDEN_DIM2 = 64
    MID_DIM = 32

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0003
    BETA_START = 0.01  # 调高熵系数以鼓励探索
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
