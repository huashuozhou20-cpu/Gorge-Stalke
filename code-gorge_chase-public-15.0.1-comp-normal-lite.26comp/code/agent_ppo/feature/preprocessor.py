#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。

特征维度扩展版本：
- 英雄自身特征: 4D
- 怪物特征: 5D × 2
- 怪物威胁向量: 4D × 2 (单位向量(dx, dz) + 倒数距离)
- 宝箱特征: 2D × 3 (最近3个宝箱的距离和方位角)
- 宝箱引导特征: 2D (最近宝箱的相对坐标)
- 局部地图特征: 49D (7×7多值掩码: 0=平地, 1=障碍, 2=怪物)
- 合法动作掩码: 8D
- 进度特征: 2D
总计: 89D
"""

import numpy as np
import math

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_DIST_BUCKET = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
LOCAL_MAP_SIZE = 7
NUM_TREASURES = 3

# 移动方向向量 (dx, dz)
DIRECTIONS = [
    (1, 0),    # 右
    (1, -1),   # 右上
    (0, -1),   # 上
    (-1, -1),  # 左上
    (-1, 0),   # 左
    (-1, 1),   # 左下
    (0, 1),    # 下
    (1, 1)     # 右下
]


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5

    def get_legal_actions(self, hero_pos, map_info):
        """实时检测8个移动方向是否有障碍物

        Args:
            hero_pos: 英雄当前位置
            map_info: 地图信息

        Returns:
            legal_action: 合法动作掩码，长度为8
        """
        legal_action = [1] * 8
        if map_info is None or len(map_info) == 0:
            return legal_action

        try:
            # 计算地图单元格大小
            cell_size = MAP_SIZE / len(map_info)
            # 计算英雄在地图网格中的位置
            hero_grid_x = int(hero_pos["x"] / cell_size)
            hero_grid_z = int(hero_pos["z"] / cell_size)

            # 检查每个方向
            for i, (dx, dz) in enumerate(DIRECTIONS):
                # 计算目标位置
                target_grid_x = hero_grid_x + dx
                target_grid_z = hero_grid_z + dz
                # 检查是否在地图范围内
                if 0 <= target_grid_x < len(map_info) and 0 <= target_grid_z < len(map_info[0]):
                    # 检查是否有障碍物（1表示障碍物）
                    if map_info[target_grid_x][target_grid_z] == 1:
                        legal_action[i] = 0
                else:
                    # 超出地图范围
                    legal_action[i] = 0
        except Exception as e:
            # 如果计算失败，返回全1掩码
            pass

        return legal_action

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

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        # 怪物威胁向量特征 (4D x 2)：单位向量(dx, dz) + 倒数距离
        monster_threat_feats = []
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

                    # 计算单位向量和倒数距离
                    if raw_dist > 0:
                        unit_dx = (m_pos["x"] - hero_pos["x"]) / raw_dist
                        unit_dz = (m_pos["z"] - hero_pos["z"]) / raw_dist
                        inv_dist = 1.0 / (raw_dist + 1e-6)  # 加小量避免除零
                        inv_dist_norm = _norm(inv_dist, 1.0 / 1.0)  # 距离为1时倒数为1
                    else:
                        unit_dx = 0.0
                        unit_dz = 0.0
                        inv_dist_norm = 1.0

                    # 添加威胁向量特征
                    monster_threat_feats.append(np.array([unit_dx, unit_dz, inv_dist_norm, is_in_view], dtype=np.float32))
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                    # 不可见时威胁向量为0
                    monster_threat_feats.append(np.zeros(4, dtype=np.float32))
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))
                monster_threat_feats.append(np.zeros(4, dtype=np.float32))

        # Treasure features (2D x 3) / 宝箱特征：最近3个宝箱的距离和方位角
        treasures = frame_state.get("treasures", [])
        treasure_dists = []
        for t in treasures:
            t_pos = t.get("pos", None)
            if t_pos is not None:
                raw_dist = np.sqrt((hero_pos["x"] - t_pos["x"]) ** 2 + (hero_pos["z"] - t_pos["z"]) ** 2)
                dx = t_pos["x"] - hero_pos["x"]
                dz = t_pos["z"] - hero_pos["z"]
                azimuth = math.atan2(dz, dx)
                treasure_dists.append((raw_dist, azimuth, t, t_pos))
        treasure_dists.sort(key=lambda x: x[0])
        treasure_feats = []
        # 宝箱引导特征 (2D)：最近宝箱的相对坐标
        nearest_treasure_feat = np.zeros(2, dtype=np.float32)
        if treasure_dists:
            _, _, _, nearest_treasure_pos = treasure_dists[0]
            # 计算相对坐标并归一化
            rel_x = nearest_treasure_pos["x"] - hero_pos["x"]
            rel_z = nearest_treasure_pos["z"] - hero_pos["z"]
            nearest_treasure_feat[0] = _norm(rel_x, MAP_SIZE)
            nearest_treasure_feat[1] = _norm(rel_z, MAP_SIZE)

        for i in range(NUM_TREASURES):
            if i < len(treasure_dists):
                raw_dist, azimuth, _, _ = treasure_dists[i]
                dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                azimuth_norm = (azimuth + math.pi) / (2 * math.pi)
                treasure_feats.append(np.array([dist_norm, azimuth_norm], dtype=np.float32))
            else:
                treasure_feats.append(np.array([1.0, 0.5], dtype=np.float32))
        treasure_feat = np.concatenate(treasure_feats)

        # Local map features (49D, 7x7 multi-value mask) / 局部地图特征：7x7多值掩码
        # 0=平地, 1=障碍, 2=怪物
        map_feat = np.zeros(LOCAL_MAP_SIZE * LOCAL_MAP_SIZE, dtype=np.float32)
        if map_info is not None and len(map_info) >= 15:
            center = len(map_info) // 2
            half_size = LOCAL_MAP_SIZE // 2
            flat_idx = 0
            for row in range(center - half_size, center + half_size + 1):
                for col in range(center - half_size, center + half_size + 1):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col])
                    flat_idx += 1
        monster_positions = []
        for m in monsters:
            if m.get("is_in_view", 0):
                m_pos = m.get("pos", None)
                if m_pos is not None:
                    monster_positions.append((m_pos["x"], m_pos["z"]))
        if map_info is not None and monster_positions:
            center = len(map_info) // 2
            half_size = LOCAL_MAP_SIZE // 2
            cell_size = MAP_SIZE / len(map_info)
            for mx, mz in monster_positions:
                rel_x = mx - hero_pos["x"]
                rel_z = mz - hero_pos["z"]
                grid_x = int(rel_x / cell_size + half_size)
                grid_z = int(rel_z / cell_size + half_size)
                if 0 <= grid_x < LOCAL_MAP_SIZE and 0 <= grid_z < LOCAL_MAP_SIZE:
                    idx = grid_z * LOCAL_MAP_SIZE + grid_x
                    if idx < len(map_feat):
                        map_feat[idx] = 2.0

        # Legal action mask (8D) / 合法动作掩码
        # 首先使用环境提供的合法动作
        legal_action = [1] * 8
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 8}
                legal_action = [1 if j in valid_set else 0 for j in range(8)]

        # 然后使用实时障碍物检测进行修正
        realtime_legal_action = self.get_legal_actions(hero_pos, map_info)
        # 取交集，只有两个都认为合法的动作才是合法的
        for i in range(8):
            legal_action[i] = legal_action[i] & realtime_legal_action[i]

        if sum(legal_action) == 0:
            legal_action = [1] * 8

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        # 4D hero + 5D*2 monster + 4D*2 monster_threat + 6D treasure + 2D nearest_treasure + 49D map + 8D legal + 2D progress
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                monster_threat_feats[0],
                monster_threat_feats[1],
                treasure_feat,
                nearest_treasure_feat,
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
