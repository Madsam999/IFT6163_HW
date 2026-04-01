"""
Generate composite figure for HW3 report:
  Top row: sampled frames from dense PPO seed0 episode
  Bottom : V(s) and shaped reward r(s) on those sampled states

Usage:
    cd /project/60004/samuel/IFT6163_HW
    python hw3/make_composite_figure.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

# hydra is imported by libero_env_fast but not actually used at class level
try:
    import hydra
except ModuleNotFoundError:
    from unittest.mock import MagicMock
    sys.modules['hydra'] = MagicMock()
    sys.modules['omegaconf'] = MagicMock()

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── replicate model definitions ──────────────────────────────────────────────

class DensePolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        features = self.net(obs)
        mean = self.mean_head(features)
        std = self.log_std.clamp(-4, 2).exp().expand_as(mean)
        from torch.distributions import Normal
        return Normal(mean, std)

    def get_action(self, obs, deterministic=True):
        dist = self.forward(obs)
        action = dist.mean if deterministic else dist.rsample()
        return action.clamp(-1.0, 1.0)


class DenseValueFunction(nn.Module):
    def __init__(self, obs_dim, hidden_dim=256, n_layers=3):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs):
        return self.value_head(self.net(obs)).squeeze(-1)


# ── load checkpoint ───────────────────────────────────────────────────────────

CKPT = 'outputs/hw3_dense_ppo_seed0/dense_ppo_100096.pth'
device = torch.device('cpu')

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
policy = DensePolicy(13, 7, 256, 3).to(device)
value_fn = DenseValueFunction(13, 256, 3).to(device)
policy.load_state_dict(ckpt['policy'])
value_fn.load_state_dict(ckpt['value_fn'])
policy.eval()
value_fn.eval()
print("Loaded checkpoint from", CKPT)

# ── run one episode ───────────────────────────────────────────────────────────

from hw3.libero_env_fast import FastLIBEROEnv

env = FastLIBEROEnv(
    benchmark_name='libero_spatial',
    task_id=9,
    max_episode_steps=600,
    render_mode='rgb_array',
    num_sim_steps=1,
)

from libero.libero import benchmark
task_suite = benchmark.get_benchmark_dict()['libero_spatial']()
init_states = task_suite.get_task_init_states(9)
env.reset()
env.set_init_state(init_states[0])
obs, info = env.reset()

frames = []
rewards = []
values = []

done, truncated, t = False, False, 0
with torch.no_grad():
    while not (done or truncated or t >= 600):
        state = np.asarray(info.get('state_obs', obs), dtype=np.float32)
        obs_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = policy.get_action(obs_t, deterministic=True).squeeze(0).numpy()
        v = float(value_fn(obs_t).item())

        obs, reward, done, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)
        rewards.append(reward)
        values.append(v)
        t += 1

env.close()
print(f"Episode length: {t}, total reward: {sum(rewards):.2f}")

# ── make figure ───────────────────────────────────────────────────────────────

N_FRAMES = 9
indices = np.linspace(0, len(frames) - 1, N_FRAMES, dtype=int)
sampled_frames = [frames[i] for i in indices]

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(2, 1, height_ratios=[1.5, 1.8], hspace=0.35)

# Top: sampled frames
ax_top = fig.add_subplot(gs[0])
ax_top.axis('off')
frame_h = sampled_frames[0].shape[0]
frame_w = sampled_frames[0].shape[1]
strip = np.concatenate(sampled_frames, axis=1)
ax_top.imshow(strip)
ax_top.set_title('Sampled Frames — Dense PPO Seed 0 (100k steps)', fontsize=11)

# Bottom: V(s) and r(s)
ax_bot = fig.add_subplot(gs[1])
timesteps = np.arange(len(rewards))
ax_bot.plot(timesteps, rewards, color='steelblue', lw=1.2, label='Shaped reward $r(s_t)$')
ax2 = ax_bot.twinx()
ax2.plot(timesteps, values, color='tomato', lw=1.4, linestyle='--', label='Value $V(s_t)$')
ax_bot.set_xlabel('Timestep', fontsize=10)
ax_bot.set_ylabel('Shaped reward $r(s_t)$', color='steelblue', fontsize=10)
ax2.set_ylabel('Value estimate $V(s_t)$', color='tomato', fontsize=10)
ax_bot.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='tomato')
# combined legend
lines1, labels1 = ax_bot.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax_bot.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)
ax_bot.set_title('Trained Value Function and Shaped Reward on Episode States', fontsize=11)

plt.savefig('hw3/composite_figure.png', dpi=150, bbox_inches='tight')
plt.savefig('hw3/composite_figure.pdf', bbox_inches='tight')
print("Saved hw3/composite_figure.png and .pdf")
