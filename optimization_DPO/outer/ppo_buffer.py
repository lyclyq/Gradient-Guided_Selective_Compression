# outer/ppo_buffer.py
import torch

class PPOBuffer:
    """
    最小 PPO 轨迹缓存：
    - logp_old: 动作采样时的 logprob
    - rewards: Val2 奖励
    目前支持单步 advantage 计算；接口兼容后续扩展
    """
    def __init__(self, momentum: float = 0.9, device=None):
        self.logp_old = []
        self.rewards = []
        self.b = torch.tensor(0.0, device=device)
        self.m = momentum
        self.device = device

    def add(self, logp_old, reward):
        if not torch.is_tensor(logp_old):
            logp_old = torch.tensor(float(logp_old), device=self.device)
        self.logp_old.append(logp_old.detach())
        self.rewards.append(float(reward))

    def advantages(self):
        if not self.rewards:
            return torch.tensor([0.0], device=self.device)
        r = torch.tensor(self.rewards, device=self.device)
        self.b = self.m * self.b + (1 - self.m) * r.mean()
        adv = r - self.b.detach()
        return adv

    def clear(self):
        self.logp_old.clear()
        self.rewards.clear()
