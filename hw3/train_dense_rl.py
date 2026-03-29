"""
HW3 Part 1: Train a dense MLP policy with PPO on privileged state.

Usage:
    python hw3/train_dense_rl.py \
        experiment.name=hw3_dense_ppo_seed0 \
        r_seed=0 \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9] \
        training.total_env_steps=200000 \
        training.rollout_length=128 \
        training.ppo_epochs=10 \
        training.minibatch_size=256
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv


# ---------------------------------------------------------------------------
# Policy and value networks
# ---------------------------------------------------------------------------

class DensePolicy(nn.Module):
    """
    MLP policy that maps privileged state observations to action distributions.
    Outputs a Gaussian distribution (mean + log_std) over the 7-DoF action space.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: (B, obs_dim) float tensor
        Returns:
            dist: torch.distributions.Normal over actions
        """
        features = self.net(obs)
        mean = self.mean_head(features)
        std = self.log_std.clamp(-4, 2).exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample an action and return (action, log_prob, entropy)."""
        dist = self.forward(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy


class DenseValueFunction(nn.Module):
    """MLP value function V(s) for PPO critic."""
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns scalar value estimate of shape (B,)."""
        return self.value_head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# PPO rollout buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores a fixed-length on-policy rollout for PPO updates."""

    def __init__(self, rollout_length: int, obs_dim: int, action_dim: int, device: torch.device):
        self.rollout_length = rollout_length
        self.device = device
        self.obs = torch.zeros(rollout_length, obs_dim, device=device)
        self.actions = torch.zeros(rollout_length, action_dim, device=device)
        self.log_probs = torch.zeros(rollout_length, device=device)
        self.rewards = torch.zeros(rollout_length, device=device)
        self.values = torch.zeros(rollout_length, device=device)
        self.dones = torch.zeros(rollout_length, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1

    def full(self):
        return self.ptr >= self.rollout_length

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(self, last_value: torch.Tensor, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages.

        Args:
            last_value: bootstrap value V(s_T) of shape ()
            gamma: discount factor
            gae_lambda: GAE lambda
        Returns:
            returns: (rollout_length,) tensor
            advantages: (rollout_length,) tensor
        """
        returns = torch.zeros_like(self.rewards)
        advantages = torch.zeros_like(self.rewards)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(self.rollout_length)):
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value = self.values[t]
        returns = advantages + self.values
        return returns, advantages


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(policy: DensePolicy,
               value_fn: DenseValueFunction,
               policy_optimizer: torch.optim.Optimizer,
               value_optimizer: torch.optim.Optimizer,
               buffer: RolloutBuffer,
               returns: torch.Tensor,
               advantages: torch.Tensor,
               cfg: DictConfig):
    """
    Perform `ppo_epochs` passes of minibatch PPO updates on the stored rollout.

    Returns a dict of mean losses for logging.
    """
    obs = buffer.obs
    actions = buffer.actions
    old_log_probs = buffer.log_probs.detach()

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # Do NOT normalize returns — the value function must learn raw return scale.
    # Normalizing per-rollout changes the target scale each update, preventing convergence.

    n = buffer.rollout_length
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(cfg.training.ppo_epochs):
        perm = torch.randperm(n, device=buffer.device)
        for start in range(0, n, cfg.training.minibatch_size):
            mb_idx = perm[start:start + cfg.training.minibatch_size]

            dist = policy.forward(obs[mb_idx])
            new_log_probs = dist.log_prob(actions[mb_idx]).sum(-1)
            entropy = dist.entropy().sum(-1).mean()

            ratio = (new_log_probs - old_log_probs[mb_idx]).exp().clamp(max=10.0)
            mb_adv = advantages[mb_idx]
            surr1 = ratio * mb_adv
            surr2 = ratio.clamp(1 - cfg.training.clip_eps, 1 + cfg.training.clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(value_fn(obs[mb_idx]), returns[mb_idx])

            # Policy update
            p_loss = policy_loss - cfg.training.entropy_coeff * entropy
            policy_optimizer.zero_grad()
            p_loss.backward(retain_graph=True)
            for p in policy.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.max_grad_norm)
            policy_optimizer.step()

            # Value function update (no grad clipping — let it converge to raw returns fast)
            v_loss = cfg.training.value_coeff * value_loss
            value_optimizer.zero_grad()
            v_loss.backward()
            for p in value_fn.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            nn.utils.clip_grad_norm_(value_fn.parameters(), cfg.training.max_grad_norm)
            value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / n_updates,
        "value_loss": total_value_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="dense_ppo", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Logging ---
    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # --- Environment ---
    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(
        task_id=task_id,
        max_episode_steps=cfg.sim.episode_length,
        cfg=cfg,
    )

    obs_dim = cfg.policy.obs_dim
    action_dim = cfg.policy.action_dim

    # --- Models ---
    policy = DensePolicy(obs_dim, action_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)
    value_fn = DenseValueFunction(obs_dim, cfg.policy.hidden_dim, cfg.policy.n_layers).to(device)

    policy_optimizer = torch.optim.Adam(
        policy.parameters(), lr=cfg.training.learning_rate,
    )
    value_optimizer = torch.optim.Adam(
        value_fn.parameters(), lr=cfg.training.learning_rate,
    )

    buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)

    # --- Rollout state ---
    obs, _ = env.reset()
    obs_tensor = torch.tensor(np.nan_to_num(obs, nan=0.0), dtype=torch.float32, device=device)
    episode_return = 0.0
    episode_steps = 0
    total_steps = 0
    episode_returns = []
    episode_successes = []

    # --- Main loop ---
    while total_steps < cfg.training.total_env_steps:
        buffer.reset()

        # Collect one rollout
        with torch.no_grad():
            for _ in range(cfg.training.rollout_length):
                action, log_prob, _ = policy.get_action(obs_tensor.unsqueeze(0))
                value = value_fn(obs_tensor.unsqueeze(0))
                action_np = action.squeeze(0).cpu().numpy()

                next_obs, reward, done, truncated, info = env.step(action_np)
                episode_return += reward
                episode_steps += 1
                total_steps += 1

                # Handle truncation: bootstrap value for correct GAE returns.
                # In this env, ALL episode endings are truncations (time limit),
                # never true terminals. Without this, GAE sees return=reward at
                # the last step instead of return=reward + γ·V(s_next), an error
                # of ~47 that corrupts advantages and causes policy degradation.
                store_reward = reward
                if truncated:
                    next_obs_tensor = torch.tensor(
                        np.nan_to_num(next_obs, nan=0.0),
                        dtype=torch.float32, device=device,
                    )
                    bootstrap_val = value_fn(next_obs_tensor.unsqueeze(0)).squeeze(0)
                    store_reward += cfg.training.gamma * bootstrap_val.item()

                buffer.add(
                    obs_tensor,
                    action.squeeze(0),
                    log_prob.squeeze(0),
                    torch.tensor(store_reward, device=device),
                    value.squeeze(0),
                    torch.tensor(float(done or truncated), device=device),
                )

                if done or truncated:
                    episode_returns.append(episode_return)
                    episode_successes.append(float(info.get("success_placed", 0.0)))
                    episode_return = 0.0
                    episode_steps = 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                obs_tensor = torch.tensor(np.nan_to_num(obs, nan=0.0), dtype=torch.float32, device=device)

                if buffer.full():
                    break

            # Bootstrap last value
            last_value = value_fn(obs_tensor.unsqueeze(0)).squeeze(0)

        returns, advantages = buffer.compute_returns_and_advantages(
            last_value, cfg.training.gamma, cfg.training.gae_lambda
        )

        # PPO update
        update_info = ppo_update(policy, value_fn, policy_optimizer, value_optimizer, buffer, returns, advantages, cfg)

        # Logging
        if total_steps % cfg.log_interval < cfg.training.rollout_length:
            log_dict = {
                "train/total_steps": total_steps,
                **{f"train/{k}": v for k, v in update_info.items()},
            }
            if episode_returns:
                log_dict["train/episode_return"] = np.mean(episode_returns[-10:])
                log_dict["train/success_rate"] = np.mean(episode_successes[-10:])
            wandb.log(log_dict, step=total_steps)
            print(f"[{total_steps}/{cfg.training.total_env_steps}] "
                  f"return={log_dict.get('train/episode_return', float('nan')):.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

        # Checkpoint
        if total_steps % cfg.save_interval < cfg.training.rollout_length:
            ckpt = {
                "policy": policy.state_dict(),
                "value_fn": value_fn.state_dict(),
                "policy_optimizer": policy_optimizer.state_dict(),
                "value_optimizer": value_optimizer.state_dict(),
                "total_steps": total_steps,
                "cfg": OmegaConf.to_container(cfg),
            }
            torch.save(ckpt, f"dense_ppo_{total_steps}.pth")

    # Final save
    torch.save({"policy": policy.state_dict(), "cfg": OmegaConf.to_container(cfg)},
               "dense_ppo_final.pth")
    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
