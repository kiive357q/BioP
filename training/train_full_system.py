# -*- coding: utf-8 -*-
"""
Full System Training - 完整系统训练脚本
BioP Causal WorldModel V2.0 - 端到端训练

【训练目标】
1. 世界模型学习动力学 (NCDE)
2. Actor-Critic学习控制策略 (Lexicographic SAC)
3. CBF学习安全边界

【仿真环境】
- BSM1FullSimulation: 完整污水处理仿真
- 传感器噪声、时序依赖、能耗模型

【训练配置】
- 序列长度: 100步
- 批处理: 32
- 训练轮数: 1000
- 设备: CUDA GPU

【版本】V2.0-FullTraining
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.bsm1_full_simulation import BioPSimulator, BSM1FullConfig, create_simulator
from models.expanded_ncde import BioPWorldModel, create_expanded_model
from safety.full_control_module import LexicographicSAC, create_full_control_system, ReplayBuffer, ControlConfig


@dataclass
class FullTrainingConfig:
    """完整训练配置"""
    # 仿真
    simulation_steps: int = 100  # 每回合步数
    n_episodes: int = 1000  # 训练回合数
    eval_interval: int = 10  # 评估间隔
    
    # 模型
    obs_dim: int = 48
    action_dim: int = 4
    latent_dim: int = 1024
    
    # 训练
    batch_size: int = 32
    replay_capacity: int = 100000
    update_interval: int = 1  # 每隔几步更新一次
    target_update_interval: int = 1
    
    # 学习率
    world_model_lr: float = 1e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    
    # 早停
    early_stopping_patience: int = 50
    early_stopping_delta: float = 0.001
    
    # 保存
    save_dir: str = "./checkpoints_full"
    log_dir: str = "./logs_full"
    
    # 设备
    device: str = "cuda"


class FullSystemTrainer:
    """
    完整系统训练器
    
    【训练流程】
    1. 仿真器生成真实数据
    2. 世界模型学习动力学
    3. 控制agent学习策略
    4. CBF学习安全边界
    """
    
    def __init__(self, config: FullTrainingConfig = None):
        self.config = config or FullTrainingConfig()
        
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # 创建目录
        Path(self.config.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._init_components()
        
        # 统计
        self.best_reward = float('-inf')
        self.episode_count = 0
        self.step_count = 0
        
        # Tensorboard
        self.writer = SummaryWriter(self.config.log_dir)
        
        # 指标记录
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'world_model_losses': [],
            'actor_losses': [],
            'critic_losses': [],
            'cbf_violations': [],
            'energy_consumption': [],
        }
    
    def _init_components(self):
        """初始化所有组件"""
        print("=" * 70)
        print("Initializing Full System")
        print("=" * 70)
        
        # 1. 仿真器
        print("\n[1/4] Creating simulator...")
        self.simulator = create_simulator(
            batch_size=self.config.batch_size,
            device=self.config.device
        )
        print(f"  State dim: {self.simulator.get_state_dim()}")
        print(f"  Action dim: {self.simulator.get_action_dim()}")
        
        # 2. 世界模型
        print("\n[2/4] Creating world model...")
        self.world_model = create_expanded_model(
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            latent_dim=self.config.latent_dim,
            device=self.config.device
        )
        
        # 世界模型优化器
        self.world_model_optimizer = torch.optim.Adam(
            [
                {'params': self.world_model.encoder.parameters()},
                {'params': self.world_model.dynamics.parameters()},
                {'params': self.world_model.decoder.parameters()},
            ],
            lr=self.config.world_model_lr
        )
        
        # 3. 控制agent
        print("\n[3/4] Creating control system...")
        self.control = create_full_control_system(
            self.world_model,
            device=self.config.device
        )
        
        # 4. 回放缓冲区
        print("\n[4/4] Creating replay buffer...")
        self.replay_buffer = ReplayBuffer(
            capacity=self.config.replay_capacity,
            obs_dim=self.config.obs_dim,
            action_dim=self.config.action_dim,
            batch_size=self.config.batch_size,
            device=self.config.device
        )
        
        print("\n✓ All components initialized")
    
    def collect_experience(self, n_steps: int, exploration: bool = True):
        """
        收集经验
        
        Args:
            n_steps: 收集步数
            exploration: 是否探索
        """
        obs = self.simulator.reset()
        episode_reward = 0.0
        episode_length = 0
        energy_total = 0.0
        
        for step in range(n_steps):
            # 选择动作
            obs_tensor = obs['observation']
            action, action_info = self.control.select_action(
                obs_tensor,
                deterministic=not exploration,
                use_cbf=True
            )
            
            # 执行
            next_obs, reward, done, info = self.simulator.step(action)
            
            # 存储
            self.replay_buffer.push(
                obs=obs_tensor[0].detach(),
                action=action[0].detach(),
                reward=reward.unsqueeze(0).detach(),
                next_obs=next_obs['observation'][0].detach(),
                done=torch.tensor([done], device=self.device)
            )
            
            episode_reward += reward.item()
            episode_length += 1
            energy_total += info.get('energy', 0)
            
            obs = next_obs
            self.step_count += 1
            
            if done:
                obs = self.simulator.reset()
        
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'energy_total': energy_total,
        }
    
    def update_world_model(self, batch: Dict) -> Dict:
        """
        更新世界模型
        
        使用MSE损失预测下一状态
        """
        obs = batch['obs']
        action = batch['action']
        next_obs = batch['next_obs']
        
        # 预测
        pred_next_obs, _ = self.world_model.predict_next_obs(
            obs, action, time_delta=0.01
        )
        
        # MSE损失
        world_model_loss = F.mse_loss(pred_next_obs, next_obs)
        
        # 更新
        self.world_model_optimizer.zero_grad()
        world_model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), 100)
        self.world_model_optimizer.step()
        
        return {'world_model_loss': world_model_loss.item()}
    
    def train_episode(self) -> Dict:
        """
        训练一个回合
        
        Returns:
            stats: 训练统计
        """
        # 1. 收集经验
        collect_stats = self.collect_experience(
            self.config.simulation_steps,
            exploration=True
        )
        
        # 2. 更新世界模型
        if len(self.replay_buffer) >= self.config.batch_size:
            batch = self.replay_buffer.sample()
            wm_stats = self.update_world_model(batch)
            
            # 3. 更新控制agent
            if self.step_count % self.config.update_interval == 0:
                control_stats = self.control.update(batch, self.step_count)
            else:
                control_stats = {}
            
            # 4. 更新目标网络
            if self.step_count % self.config.target_update_interval == 0:
                self.world_model.sync_target_networks()
        else:
            wm_stats = {}
            control_stats = {}
        
        # 组合统计
        stats = {
            **collect_stats,
            **wm_stats,
            **control_stats,
        }
        
        return stats
    
    def evaluate(self, n_episodes: int = 5) -> Dict:
        """评估性能"""
        eval_rewards = []
        eval_lengths = []
        eval_energy = []
        eval_constraints = []
        
        for _ in range(n_episodes):
            obs = self.simulator.reset()
            episode_reward = 0.0
            episode_length = 0
            energy_total = 0.0
            n_violations = 0
            
            for _ in range(self.config.simulation_steps):
                obs_tensor = obs['observation']
                
                # 确定性动作
                action, action_info = self.control.select_action(
                    obs_tensor,
                    deterministic=True,
                    use_cbf=True
                )
                
                # 执行
                next_obs, reward, done, info = self.simulator.step(action)
                
                episode_reward += reward.item()
                episode_length += 1
                energy_total += info.get('energy', 0)
                
                if not action_info.get('safe', True):
                    n_violations += 1
                
                obs = next_obs
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_energy.append(energy_total)
            eval_constraints.append(n_violations)
        
        return {
            'eval_reward_mean': sum(eval_rewards) / len(eval_rewards),
            'eval_length_mean': sum(eval_lengths) / len(eval_lengths),
            'eval_energy_mean': sum(eval_energy) / len(eval_energy),
            'eval_constraints_mean': sum(eval_constraints) / len(eval_constraints),
        }
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint = {
            'world_model': self.world_model.state_dict(),
            'control': {
                'actor': self.control.actor.state_dict(),
                'critic': self.control.critic.state_dict(),
                'cbf': self.control.cbf.state_dict(),
            },
            'optimizer': {
                'world_model': self.world_model_optimizer.state_dict(),
            },
            'config': asdict(self.config),
            'best_reward': self.best_reward,
            'episode_count': self.episode_count,
        }
        
        path = Path(self.config.save_dir) / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"  ✓ Saved checkpoint: {path}")
    
    def train(self):
        """
        主训练循环
        """
        print("\n" + "=" * 70)
        print("Starting Full System Training")
        print("=" * 70)
        
        print(f"\n配置:")
        print(f"  设备: {self.device}")
        print(f"  训练回合: {self.config.n_episodes}")
        print(f"  每回合步数: {self.config.simulation_steps}")
        print(f"  批处理: {self.config.batch_size}")
        print(f"  序列维度: {self.config.latent_dim}")
        
        start_time = time.time()
        
        for episode in range(self.config.n_episodes):
            self.episode_count = episode
            
            # 训练
            stats = self.train_episode()
            
            # 记录指标
            self.metrics['episode_rewards'].append(stats.get('episode_reward', 0))
            self.metrics['episode_lengths'].append(stats.get('episode_length', 0))
            self.metrics['world_model_losses'].append(stats.get('world_model_loss', 0))
            self.metrics['actor_losses'].append(stats.get('actor_loss', 0))
            self.metrics['critic_losses'].append(stats.get('critic_loss', 0))
            
            # Tensorboard
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'train/{key}', value, episode)
            
            # 评估
            if episode % self.config.eval_interval == 0:
                eval_stats = self.evaluate(n_episodes=3)
                
                # Tensorboard
                for key, value in eval_stats.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'eval/{key}', value, episode)
                
                # 保存最佳
                if eval_stats['eval_reward_mean'] > self.best_reward:
                    self.best_reward = eval_stats['eval_reward_mean']
                    self.save_checkpoint('best_model')
                
                # 打印
                elapsed = time.time() - start_time
                print(f"\n[Episode {episode}/{self.config.n_episodes}] "
                      f"({elapsed:.0f}s elapsed)")
                print(f"  Eval Reward: {eval_stats['eval_reward_mean']:.2f}")
                print(f"  Eval Length: {eval_stats['eval_length_mean']:.1f}")
                print(f"  Eval Energy: {eval_stats['eval_energy_mean']:.2f}")
                print(f"  Eval Constraints: {eval_stats['eval_constraints_mean']:.1f}")
                print(f"  WMLoss: {stats.get('world_model_loss', 0):.4f}")
                print(f"  ActorLoss: {stats.get('actor_loss', 0):.4f}")
            
            # 定期保存
            if episode % 100 == 0:
                self.save_checkpoint(f'episode_{episode}')
        
        # 保存最终模型
        self.save_checkpoint('final_model')
        
        # 保存指标
        metrics_path = Path(self.config.log_dir) / 'metrics.json'
        with open(metrics_path, 'w') as f:
            # 转换非JSON类型
            metrics_save = {}
            for k, v in self.metrics.items():
                if all(isinstance(x, (int, float)) for x in v):
                    metrics_save[k] = v
            json.dump(metrics_save, f, indent=2)
        
        print(f"\n✓ Training completed!")
        print(f"  Total time: {time.time() - start_time:.0f}s")
        print(f"  Best reward: {self.best_reward:.2f}")
        print(f"  Saved to: {self.config.save_dir}")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        wm_info = self.world_model.get_state_dict_info()
        
        return {
            'world_model_params': wm_info['total'],
            'latent_dim': self.config.latent_dim,
            'obs_dim': self.config.obs_dim,
            'action_dim': self.config.action_dim,
            'simulation_steps': self.config.simulation_steps,
        }


def main():
    """主函数"""
    print("=" * 70, flush=True)
    print("BioP Causal WorldModel V2.0 - Full System Training", flush=True)
    print("=" * 70, flush=True)
    
    # 训练配置
    config = FullTrainingConfig(
        simulation_steps=100,
        n_episodes=1000,
        batch_size=32,
        latent_dim=1024,
        obs_dim=48,
        action_dim=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 创建训练器
    trainer = FullSystemTrainer(config)
    
    # 打印模型信息
    info = trainer.get_model_info()
    print("\n模型信息:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main()
