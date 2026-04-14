#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def calculate_relative_direction(hero_pos, target_pos):
    """Calculate relative direction from hero to target.

    计算英雄到目标的相对方向。
    返回值：方向索引 (1-8)，对应8个方向
    1: 北, 2: 东北, 3: 东, 4: 南, 5: 西南, 6: 西, 7: 西北, 8: 东南
    """
    dx = target_pos[0] - hero_pos[0]
    dz = target_pos[1] - hero_pos[1]
    
    # 特殊情况：目标与英雄位置相同
    if dx == 0 and dz == 0:
        return 1  # 默认返回北
    
    # 计算象限和方向
    if dx > 0 and dz > 0:
        # 东南方向
        return 8
    elif dx > 0 and dz < 0:
        # 东北方向
        return 2
    elif dx < 0 and dz > 0:
        # 西南方向
        return 5
    elif dx < 0 and dz < 0:
        # 西北方向
        return 7
    elif dx > 0:
        # 东方向
        return 3
    elif dx < 0:
        # 西方向
        return 6
    elif dz > 0:
        # 南方向
        return 4
    else:
        # 北方向
        return 1


def find_last_passable_cell(start_pos, target_pos, map_info):
    """Find the last passable cell along the path from start to target.

    找到从起点到目标点路径上的最后一个可通行方格。
    """
    # 计算移动方向向量
    dx = target_pos[0] - start_pos[0]
    dz = target_pos[1] - start_pos[1]
    
    # 计算距离
    distance = max(abs(dx), abs(dz))
    if distance == 0:
        return start_pos
    
    # 计算单位方向向量
    step_x = dx / distance
    step_z = dz / distance
    
    # 从起点开始，逐步向目标移动
    current_x, current_z = start_pos
    last_passable = (current_x, current_z)
    
    for i in range(1, int(distance) + 1):
        next_x = int(round(start_pos[0] + step_x * i))
        next_z = int(round(start_pos[1] + step_z * i))
        
        # 检查是否在地图范围内
        if 0 <= next_x < len(map_info) and 0 <= next_z < len(map_info[0]):
            # 检查是否可通行（map_info中0表示可通行）
            if map_info[next_x][next_z] == 0:
                last_passable = (next_x, next_z)
            else:
                # 遇到障碍物，返回上一个可通行位置
                break
        else:
            # 超出地图范围，返回上一个可通行位置
            break
    
    return last_passable


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        # 怪物位置历史记录，用于轨迹预判
        self.monster_pos_history = {}
        # 初始化每个怪物的位置历史，最多保存5帧
        for i in range(2):
            self.monster_pos_history[i] = []

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (5D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)
        # 英雄当前是否处于加速Buff状态
        is_buff_active = 1.0 if hero["buff_remaining_time"] > 0 else 0.0

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm, is_buff_active], dtype=np.float32)

        # Chest features (2D x 3) / 宝箱特征
        chests = frame_state.get("chests", [])
        chest_feats = []
        # 计算所有宝箱到英雄的距离
        chest_distances = []
        for chest in chests:
            chest_pos = chest.get("pos", {})
            if "x" in chest_pos and "z" in chest_pos:
                dx = chest_pos["x"] - hero_pos["x"]
                dz = chest_pos["z"] - hero_pos["z"]
                distance = np.sqrt(dx**2 + dz**2)
                chest_distances.append((distance, dx, dz))
        # 按距离排序，取最近的3个
        chest_distances.sort(key=lambda x: x[0])
        for i in range(3):
            if i < len(chest_distances):
                _, dx, dz = chest_distances[i]
                # 归一化相对位置
                dx_norm = _norm(dx, MAP_SIZE)  # 相对于地图大小归一化
                dz_norm = _norm(dz, MAP_SIZE)
                chest_feats.append(np.array([dx_norm, dz_norm], dtype=np.float32))
            else:
                # 没有足够的宝箱，用0填充
                chest_feats.append(np.zeros(2, dtype=np.float32))

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        monster_movement_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)

                    # 记录怪物位置历史
                    self.monster_pos_history[i].append((m_pos["x"], m_pos["z"]))
                    # 只保留最近5帧的位置
                    if len(self.monster_pos_history[i]) > 5:
                        self.monster_pos_history[i] = self.monster_pos_history[i][-5:]

                    # 计算怪物移动向量
                    if len(self.monster_pos_history[i]) >= 2:
                        # 取最近两帧的位置计算移动向量
                        prev_pos = self.monster_pos_history[i][-2]
                        curr_pos = self.monster_pos_history[i][-1]
                        dx = curr_pos[0] - prev_pos[0]
                        dz = curr_pos[1] - prev_pos[1]
                        # 归一化移动向量
                        movement_norm = np.sqrt(dx**2 + dz**2)
                        if movement_norm > 1e-6:
                            dx_norm = dx / movement_norm
                            dz_norm = dz / movement_norm
                        else:
                            dx_norm = 0.0
                            dz_norm = 0.0
                    else:
                        # 位置历史不足，用0填充
                        dx_norm = 0.0
                        dz_norm = 0.0
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    dx_norm = 0.0
                    dz_norm = 0.0
                    # 清空该怪物的位置历史
                    self.monster_pos_history[i] = []
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
                # 添加怪物移动向量特征
                monster_movement_feats.append(np.array([dx_norm, dz_norm], dtype=np.float32))
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))
                monster_movement_feats.append(np.zeros(2, dtype=np.float32))
                # 清空该怪物的位置历史
                self.monster_pos_history[i] = []

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # Legal action mask (16D) / 合法动作掩码
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        # 根据flash_cooldown状态动态开启闪现动作（索引8-15）
        flash_cooldown = hero.get("flash_cooldown", 0)
        if flash_cooldown > 0:
            # 闪现冷却中，禁用所有闪现动作
            for j in range(8, 16):
                legal_action[j] = 0

        if sum(legal_action) == 0:
            legal_action = [1] * 16

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                chest_feats[0],
                chest_feats[1],
                chest_feats[2],
                monster_movement_feats[0],
                monster_movement_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        self.last_min_monster_dist_norm = cur_min_dist_norm

        reward = [survive_reward + dist_shaping]

        return feature, legal_action, reward


def test_direction_calculation():
    """测试相对方位计算"""
    # 测试用例：英雄在(10, 10)，怪物在(12, 12)
    hero_pos = (10, 10)
    monster_pos = (12, 12)
    direction = calculate_relative_direction(hero_pos, monster_pos)
    
    # 预期方向：东南（8）
    print(f"英雄位置: {hero_pos}")
    print(f"怪物位置: {monster_pos}")
    print(f"计算出的方向索引: {direction}")
    print(f"预期方向索引: 8 (东南)")
    print(f"测试结果: {'通过' if direction == 8 else '失败'}")
    print()


def test_flash_pathfinding():
    """测试闪现落点逻辑"""
    # 创建一个简单的地图：0表示可通行，1表示障碍物
    # 地图尺寸：20x20
    map_info = [[0 for _ in range(20)] for _ in range(20)]
    
    # 在路径上设置障碍物
    # 从(10, 10)到(15, 15)的路径上设置障碍物
    for i in range(12, 15):
        map_info[i][i] = 1  # 在(12,12)到(14,14)设置障碍物
    
    # 测试用例：英雄在(10, 10)，闪现目标为(15, 15)
    start_pos = (10, 10)
    target_pos = (15, 15)
    last_passable = find_last_passable_cell(start_pos, target_pos, map_info)
    
    print(f"起点位置: {start_pos}")
    print(f"目标位置: {target_pos}")
    print(f"最后一个可通行位置: {last_passable}")
    print(f"预期位置: (11, 11)（障碍物前的最后一个可通行位置）")
    print(f"测试结果: {'通过' if last_passable == (11, 11) else '失败'}")
    print()


if __name__ == "__main__":
    print("测试相对方位计算:")
    test_direction_calculation()
    print("测试闪现落点逻辑:")
    test_flash_pathfinding()
