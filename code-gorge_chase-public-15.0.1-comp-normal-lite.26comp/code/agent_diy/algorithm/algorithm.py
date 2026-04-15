#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

PPO algorithm implementation for DIY agent.
自定义智能体 PPO 算法实现。

损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss

  - value_loss  : Clipped value function loss（裁剪价值函数损失）
  - policy_loss : PPO Clipped surrogate objective（PPO 裁剪替代目标）
  - entropy_loss: Action entropy regularization（动作熵正则化，鼓励探索）
"""

import torch
import numpy as np
from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.vf_coef = Config.VALUE_LOSS_COEFF
        self.entropy_coef = Config.ENTROPY_LOSS_COEFF

    def learn(self, list_sample_data):
        obs = np.array([s.obs for s in list_sample_data], dtype=np.float32)
        legal_actions = np.array([s.legal_actions for s in list_sample_data], dtype=np.float32)
        actions = np.array([s.actions for s in list_sample_data], dtype=np.int64)
        probs = np.array([s.probs for s in list_sample_data], dtype=np.float32)
        rewards = np.array([s.rewards for s in list_sample_data], dtype=np.float32)
        advantages = np.array([s.advantages for s in list_sample_data], dtype=np.float32)
        values = np.array([s.values for s in list_sample_data], dtype=np.float32)
        reward_sum = np.array([s.reward_sum for s in list_sample_data], dtype=np.float32)

        obs_tensor = torch.from_numpy(obs).to(self.device)
        actions_tensor = torch.from_numpy(actions).to(self.device)
        probs_tensor = torch.from_numpy(probs).to(self.device)
        advantages_tensor = torch.from_numpy(advantages).to(self.device)
        values_tensor = torch.from_numpy(values).to(self.device)
        reward_sum_tensor = torch.from_numpy(reward_sum).to(self.device)
        legal_actions_tensor = torch.from_numpy(legal_actions).to(self.device)

        self.model.set_train_mode()
        logits, value_pred = self.model(obs_tensor)

        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_tensor)
        old_log_probs = torch.log(probs_tensor.gather(1, actions_tensor))

        ratio = torch.exp(action_log_probs - old_log_probs.detach())
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_tensor
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        value_pred_clipped = values_tensor + torch.clamp(
            value_pred - values_tensor, -0.2, 0.2
        )
        value_losses = (value_pred - reward_sum_tensor) ** 2
        value_losses_clipped = (value_pred_clipped - reward_sum_tensor) ** 2
        value_loss = 0.5 * torch.mean(torch.max(value_losses, value_losses_clipped))

        probs_softmax = torch.softmax(logits, dim=-1)
        entropy = -torch.mean(torch.sum(probs_softmax * log_probs, dim=-1))

        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": total_loss.item(),
        }
