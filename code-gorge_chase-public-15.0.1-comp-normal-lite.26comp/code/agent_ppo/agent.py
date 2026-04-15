#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob = self._run_model(feature, legal_action)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False, env_obs=env_obs)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        try:
            state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
            torch.save(state_dict_cpu, model_file_path)
            self.logger.info(f"save model {model_file_path} successfully")
        except Exception as e:
            self.logger.error(f"保存模型失败: {str(e)}")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if not os.path.exists(model_file_path):
            self.logger.error(f"模型文件 {model_file_path} 不存在")
            return
        try:
            # 加载模型文件
            checkpoint = torch.load(model_file_path, map_location=self.device)
            
            # 版本兼容性检查
            if isinstance(checkpoint, dict):
                # 检查是否包含状态字典
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
                
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.error(f"模型文件 {model_file_path} 不存在")
        except torch.nn.modules.module.ModuleNotFoundError as e:
            self.logger.error(f"模型结构不匹配: {str(e)}")
        except RuntimeError as e:
            self.logger.error(f"模型加载失败（参数不匹配）: {str(e)}")
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")

    def action_process(self, act_data, is_stochastic=True, env_obs=None):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        action_id = int(action[0])
        
        # 检查是否为闪现动作且需要判断冷却
        if 8 <= action_id <= 15 and env_obs is not None:
            # 从 env_obs 中提取 talent_cooldown
            try:
                talent_cooldown = 0
                # 更具体地捕获可能的 KeyError
                observation = env_obs["observation"]
                frame_state = observation["frame_state"]
                hero = frame_state["heroes"]
                talent_cooldown = hero.get("flash_cooldown", 0)
                
                # 如果闪现冷却为 0，保持闪现动作
                if talent_cooldown == 0:
                    # 保持闪现动作
                    pass
                else:
                    # 闪现仍在冷却，降级为普通移动
                    action_id = action_id - 8
            except KeyError as e:
                # 更具体地捕获 KeyError
                if self.logger:
                    self.logger.error(f"提取闪现冷却信息时发生 KeyError: {str(e)}")
            except Exception as e:
                # 捕获其他异常
                if self.logger:
                    self.logger.error(f"提取闪现冷却信息失败: {str(e)}")
        
        self.last_action = action_id
        return action_id

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
