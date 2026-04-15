#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for DIY agent.
自定义智能体特征预处理。
"""

import numpy as np
import math

MAP_SIZE = 128.0
MAX_MONSTER_SPEED = 5.0
MAX_FLASH_CD = 2000.0
MAX_BUFF_DURATION = 50.0
LOCAL_MAP_SIZE = 7
NUM_TREASURES = 3


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5
        self.last_hero_pos = None
        self.pos_history = []

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        treasures = frame_state.get("treasures", [])
        treasure_dists = []
        for t in treasures:
            t_pos = t.get("pos", None)
            if t_pos is not None:
                raw_dist = np.sqrt((hero_pos["x"] - t_pos["x"]) ** 2 + (hero_pos["z"] - t_pos["z"]) ** 2)
                dx = t_pos["x"] - hero_pos["x"]
                dz = t_pos["z"] - hero_pos["z"]
                azimuth = math.atan2(dz, dx)
                treasure_dists.append((raw_dist, azimuth, t))
        treasure_dists.sort(key=lambda x: x[0])
        treasure_feats = []
        for i in range(NUM_TREASURES):
            if i < len(treasure_dists):
                raw_dist, azimuth, _ = treasure_dists[i]
                dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                azimuth_norm = (azimuth + math.pi) / (2 * math.pi)
                treasure_feats.append(np.array([dist_norm, azimuth_norm], dtype=np.float32))
            else:
                treasure_feats.append(np.array([1.0, 0.5], dtype=np.float32))
        treasure_feat = np.concatenate(treasure_feats)

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

        legal_action = [1] * 8
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 8}
                legal_action = [1 if j in valid_set else 0 for j in range(8)]

        if sum(legal_action) == 0:
            legal_action = [1] * 8

        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        vel_feat = np.zeros(4, dtype=np.float32)
        if self.last_hero_pos is not None:
            dx = hero_pos["x"] - self.last_hero_pos[0]
            dz = hero_pos["z"] - self.last_hero_pos[1]
            vel_feat[0] = _norm(dx, 10.0, -10.0)
            vel_feat[1] = _norm(dz, 10.0, -10.0)
            speed = np.sqrt(dx**2 + dz**2)
            vel_feat[2] = _norm(speed, 10.0)
            if speed > 1e-6:
                vel_feat[3] = (math.atan2(dz, dx) + math.pi) / (2 * math.pi)
        self.last_hero_pos = (hero_pos["x"], hero_pos["z"])

        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                treasure_feat,
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
                vel_feat,
            ]
        )

        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        self.last_min_monster_dist_norm = cur_min_dist_norm

        reward = [survive_reward + dist_shaping]

        return feature, legal_action, reward
