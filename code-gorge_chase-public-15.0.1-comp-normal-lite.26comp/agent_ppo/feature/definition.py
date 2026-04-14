#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions, GAE computation for Gorge Chase PPO.
峡谷追猎 PPO 数据类定义与 GAE 计算。
"""

import numpy as np

# 尝试导入create_cls，如果失败则定义一个简单的实现
try:
    from common_python.utils.common_func import create_cls, attached
except ImportError:
    # 简单实现create_cls函数
    def create_cls(name, **fields):
        """Create a simple class with given fields."""
        class cls:
            def __init__(self, **kwargs):
                for key, default in fields.items():
                    setattr(self, key, kwargs.get(key, default))
        cls.__name__ = name
        return cls
    
    def attached(func):
        """Simple decorator that does nothing."""
        return func

from agent_ppo.conf.conf import Config


# ObsData: feature=40D vector, legal_action=8D mask / 特征向量与合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: action, d_action(greedy), prob, value / 动作、贪心动作、概率、价值
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: single-frame sample with int dims / 单帧样本（整数表示维度）
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    """Fill next_value and compute GAE advantage.

    填充 next_value 并使用 GAE 计算优势函数。
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Compute GAE (Generalized Advantage Estimation).

    计算广义优势估计（GAE）。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value


def reward_shaping(state, _state, action, env_obs):
    """Reward shaping function.

    奖励塑形函数，实现以下奖励逻辑：
    1. 宝箱获取奖励：当 _state.treasure_collected_count > state.treasure_collected_count 时，给予 +15.0 的即时奖励
    2. 距离塑形优化：当英雄靠近宝箱时，给予微小的正奖励；当英雄进入怪物 5 格范围内时，负奖励指数级增加
    3. 闪现效率奖励：如果使用闪现后与怪物的距离显著增大，给予 +0.5 的奖励以鼓励合理交出技能
    """
    reward = 0.0
    
    # 1. 宝箱获取奖励
    if hasattr(_state, 'treasure_collected_count') and hasattr(state, 'treasure_collected_count'):
        if _state.treasure_collected_count > state.treasure_collected_count:
            reward += 15.0
    
    # 2. 距离塑形优化
    observation = env_obs.get("observation", {})
    frame_state = observation.get("frame_state", {})
    
    # 获取英雄位置
    hero = frame_state.get("heroes", {})
    hero_pos = hero.get("pos", {})
    if hero_pos:
        hero_x, hero_z = hero_pos.get("x", 0), hero_pos.get("z", 0)
        
        # 计算与最近宝箱的距离，给予微小正奖励
        chests = frame_state.get("chests", [])
        min_chest_dist = float('inf')
        for chest in chests:
            chest_pos = chest.get("pos", {})
            if chest_pos:
                chest_x, chest_z = chest_pos.get("x", 0), chest_pos.get("z", 0)
                dist = ((hero_x - chest_x) ** 2 + (hero_z - chest_z) ** 2) ** 0.5
                if dist < min_chest_dist:
                    min_chest_dist = dist
        
        # 靠近宝箱的正奖励
        if min_chest_dist < float('inf'):
            # 距离越近，奖励越高，最大奖励0.1
            chest_reward = max(0, 0.1 - min_chest_dist / 100)
            reward += chest_reward
        
        # 计算与怪物的距离，进入5格范围时负奖励指数级增加
        monsters = frame_state.get("monsters", [])
        for monster in monsters:
            if monster.get("is_in_view", False):
                m_pos = monster.get("pos", {})
                if m_pos:
                    m_x, m_z = m_pos.get("x", 0), m_pos.get("z", 0)
                    dist = ((hero_x - m_x) ** 2 + (hero_z - m_z) ** 2) ** 0.5
                    # 进入5格范围时，负奖励指数级增加
                    if dist < 5:
                        # 距离越近，负奖励越大，最大负奖励-5
                        monster_penalty = -5 * (1 - dist / 5)
                        reward += monster_penalty
    
    # 3. 闪现效率奖励
    # 检查是否使用了闪现技能（假设动作8是闪现）
    if action == 8:
        # 计算使用闪现前后与怪物的距离变化
        if hasattr(state, 'monster_distances') and hasattr(_state, 'monster_distances'):
            # 假设monster_distances存储了与每个怪物的距离
            for i, (prev_dist, curr_dist) in enumerate(zip(state.monster_distances, _state.monster_distances)):
                if curr_dist - prev_dist > 10:  # 距离增加超过10格
                    reward += 0.5
                    break  # 只奖励一次
    
    return reward


def SampleData2NumpyData(sample_data):
    """Convert SampleData to numpy array.

    将 SampleData 转换为 numpy 数组。
    """
    # 按照字段顺序构建数组
    data = np.concatenate([
        np.array(sample_data.obs, dtype=np.float32),
        np.array(sample_data.legal_action, dtype=np.float32),
        np.array(sample_data.act, dtype=np.int32),
        np.array(sample_data.reward, dtype=np.float32),
        np.array(sample_data.reward_sum, dtype=np.float32),
        np.array(sample_data.done, dtype=np.bool_),
        np.array(sample_data.value, dtype=np.float32),
        np.array(sample_data.next_value, dtype=np.float32),
        np.array(sample_data.advantage, dtype=np.float32),
        np.array(sample_data.prob, dtype=np.float32)
    ])
    return data


def NumpyData2SampleData(numpy_data):
    """Convert numpy array back to SampleData.

    将 numpy 数组转换回 SampleData。
    """
    # 计算各字段的起始和结束位置
    obs_end = Config.DIM_OF_OBSERVATION
    legal_action_end = obs_end + Config.ACTION_NUM
    act_end = legal_action_end + 1
    reward_end = act_end + Config.VALUE_NUM
    reward_sum_end = reward_end + Config.VALUE_NUM
    done_end = reward_sum_end + 1
    value_end = done_end + Config.VALUE_NUM
    next_value_end = value_end + Config.VALUE_NUM
    advantage_end = next_value_end + Config.VALUE_NUM
    prob_end = advantage_end + Config.ACTION_NUM
    
    # 提取各字段数据
    obs = numpy_data[:obs_end].tolist()
    legal_action = numpy_data[obs_end:legal_action_end].tolist()
    act = [int(numpy_data[legal_action_end])]
    reward = numpy_data[act_end:reward_end].tolist()
    reward_sum = numpy_data[reward_end:reward_sum_end].tolist()
    done = [bool(numpy_data[reward_sum_end])]
    value = numpy_data[done_end:value_end].tolist()
    next_value = numpy_data[value_end:next_value_end].tolist()
    advantage = numpy_data[next_value_end:advantage_end].tolist()
    prob = numpy_data[advantage_end:prob_end].tolist()
    
    # 创建 SampleData 实例
    sample_data = SampleData(
        obs=obs,
        legal_action=legal_action,
        act=act,
        reward=reward,
        reward_sum=reward_sum,
        done=done,
        value=value,
        next_value=next_value,
        advantage=advantage,
        prob=prob
    )
    return sample_data


def test_sample_data_conversion():
    """测试 SampleData 和 numpy 数组之间的转换"""
    # 创建一个 SampleData 实例
    import random
    
    # 生成随机特征向量
    obs = [random.random() for _ in range(Config.DIM_OF_OBSERVATION)]
    # 生成随机合法动作掩码
    legal_action = [1 if random.random() > 0.5 else 0 for _ in range(Config.ACTION_NUM)]
    # 生成随机动作
    act = [random.randint(0, Config.ACTION_NUM - 1)]
    # 生成随机奖励
    reward = [random.random() * 10 - 5]
    # 生成随机奖励和
    reward_sum = [random.random() * 20 - 10]
    # 生成随机done标志
    done = [random.random() > 0.5]
    # 生成随机价值
    value = [random.random() * 10 - 5]
    # 生成随机下一个价值
    next_value = [random.random() * 10 - 5]
    # 生成随机优势
    advantage = [random.random() * 10 - 5]
    # 生成随机概率
    prob = [random.random() for _ in range(Config.ACTION_NUM)]
    # 归一化概率
    prob_sum = sum(prob)
    prob = [p / prob_sum for p in prob]
    
    # 创建 SampleData 实例
    original_sample = SampleData(
        obs=obs,
        legal_action=legal_action,
        act=act,
        reward=reward,
        reward_sum=reward_sum,
        done=done,
        value=value,
        next_value=next_value,
        advantage=advantage,
        prob=prob
    )
    
    # 转换为 numpy 数组
    numpy_data = SampleData2NumpyData(original_sample)
    print(f"转换后的 numpy 数组形状: {numpy_data.shape}")
    
    # 转换回 SampleData
    converted_sample = NumpyData2SampleData(numpy_data)
    
    # 验证转换前后的数据一致性
    assert len(original_sample.obs) == len(converted_sample.obs), "obs 长度不一致"
    assert len(original_sample.legal_action) == len(converted_sample.legal_action), "legal_action 长度不一致"
    assert original_sample.act == converted_sample.act, "act 不一致"
    assert original_sample.done == converted_sample.done, "done 不一致"
    
    # 验证数值字段的一致性（允许微小的浮点数误差）
    for i in range(len(original_sample.obs)):
        assert abs(original_sample.obs[i] - converted_sample.obs[i]) < 1e-6, f"obs[{i}] 不一致"
    
    for i in range(len(original_sample.legal_action)):
        assert original_sample.legal_action[i] == converted_sample.legal_action[i], f"legal_action[{i}] 不一致"
    
    for i in range(len(original_sample.reward)):
        assert abs(original_sample.reward[i] - converted_sample.reward[i]) < 1e-6, f"reward[{i}] 不一致"
    
    for i in range(len(original_sample.reward_sum)):
        assert abs(original_sample.reward_sum[i] - converted_sample.reward_sum[i]) < 1e-6, f"reward_sum[{i}] 不一致"
    
    for i in range(len(original_sample.value)):
        assert abs(original_sample.value[i] - converted_sample.value[i]) < 1e-6, f"value[{i}] 不一致"
    
    for i in range(len(original_sample.next_value)):
        assert abs(original_sample.next_value[i] - converted_sample.next_value[i]) < 1e-6, f"next_value[{i}] 不一致"
    
    for i in range(len(original_sample.advantage)):
        assert abs(original_sample.advantage[i] - converted_sample.advantage[i]) < 1e-6, f"advantage[{i}] 不一致"
    
    for i in range(len(original_sample.prob)):
        assert abs(original_sample.prob[i] - converted_sample.prob[i]) < 1e-6, f"prob[{i}] 不一致"
    
    print("转换前后数据完全一致，测试通过！")


if __name__ == "__main__":
    test_sample_data_conversion()
