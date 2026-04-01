"""
HW3 Part 2: Fine-tune a transformer policy from HW1 with PPO or GRPO.

Usage (PPO):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_ppo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=ppo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]

Usage (GRPO with ground-truth resets):
    python hw3/train_transformer_rl.py \
        experiment.name=hw3_transformer_grpo_seed0 \
        r_seed=0 \
        init_checkpoint=/path/to/hw1/miniGRP.pth \
        rl.algorithm=grpo \
        sim.task_set=libero_spatial \
        sim.eval_tasks=[9]
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mini-grp'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from hw3.libero_env_fast import FastLIBEROEnv
from hw3.train_dense_rl import RolloutBuffer, ppo_update


# ---------------------------------------------------------------------------
# Separate value network (used with transformer policy)
# ---------------------------------------------------------------------------

class ValueFunction(nn.Module):
    """
    Separate MLP value network V(s).
    Keep this separate from the transformer policy as required by hw3.md.
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(obs)).squeeze(-1)


# ---------------------------------------------------------------------------
# Transformer policy wrapper
# ---------------------------------------------------------------------------

class TransformerPolicyWrapper:
    """
    Wraps the HW1 GRP transformer model to provide a gym-style action interface.
    """
    def __init__(self, checkpoint_path: str, device: torch.device, cfg: DictConfig,
                 obs_dim: int = 13):
        import dill
        from hw3.grp_model import calc_positional_embeddings
        self.calc_positional_embeddings = calc_positional_embeddings

        self.model = torch.load(checkpoint_path, map_location=device, pickle_module=dill)
        self.model.to(device)
        self.model.train()
        self.device = device
        self.cfg = cfg
        action_dim = cfg.policy.action_dim

        self.n_embd = self.model.class_tokens.shape[-1]
        self.model.rl_log_std = nn.Parameter(torch.zeros(action_dim, device=device))

        pose_in_dim = self.model.lin_map_pose.in_features if hasattr(self.model.lin_map_pose, 'in_features') else None
        print(f"[INFO] Model input_d={self.n_embd}, lin_map_pose input={pose_in_dim}, obs_dim={obs_dim}")

        if pose_in_dim is not None and pose_in_dim != obs_dim:
            self.pose_adapter = nn.Linear(obs_dim, pose_in_dim).to(device)
        else:
            self.pose_adapter = None

    def reset_context(self):
        pass

    def _compute_action_mean(self, obs_t: torch.Tensor) -> torch.Tensor:
        B = obs_t.shape[0]
        model = self.model
        action_dim = self.cfg.policy.action_dim

        if self.pose_adapter is not None:
            pose_input = self.pose_adapter(obs_t)
        else:
            pose_input = obs_t
        state_emb = model.lin_map_pose(pose_input).unsqueeze(1)

        cls_tokens = model.class_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, state_emb], dim=1)

        seq_len = x.shape[1]
        pos_emb = self.calc_positional_embeddings(seq_len, self.n_embd)
        pos_emb = pos_emb.to(self.device)
        x = x + pos_emb.unsqueeze(0)

        for block in model.blocks:
            x = block(x)

        x = model.ln_f(x)
        cls_out = x[:, 0, :]
        raw = model.mlp(cls_out)

        return raw.reshape(B, -1)[:, :action_dim]

    def forward(self, obs: torch.Tensor):
        mean = self._compute_action_mean(obs)
        std = self.model.rl_log_std.clamp(-4, 2).exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)

        dist = self.forward(obs_t)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        # FIX: Do not clamp the action here. Calculate log_prob on the raw, 
        # unbounded action so the math checks out during PPO/GRPO updates.
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return (
            action.squeeze(0).cpu().detach().numpy(),
            log_prob.squeeze(0).detach(),
            entropy.squeeze(0).detach(),
        )

    def parameters(self):
        import itertools
        params = [self.model.parameters()]
        if self.pose_adapter is not None:
            params.append(self.pose_adapter.parameters())
        return itertools.chain(*params)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


# ---------------------------------------------------------------------------
# GRPO helpers
# ---------------------------------------------------------------------------

def collect_grpo_group(env: FastLIBEROEnv,
                       policy: TransformerPolicyWrapper,
                       init_state,
                       group_size: int,
                       max_steps: int,
                       device: torch.device):
    trajectories = []
    for _ in range(group_size):
        env.env.reset()
        env.env.sim.set_state(init_state)
        env.env.sim.forward()
        env.current_step = 0
        
        # FIX: Grab the observation directly without executing a settle step 
        # that might force the gripper open.
        settle_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs_dict = env.env.step(settle_action)[0] 
        obs = env._get_state_obs(obs_dict)
        policy.reset_context()

        traj = {
            "obs": [], "actions": [], "log_probs": [],
            "rewards": [], "dones": [], "total_return": 0.0,
        }

        for _ in range(max_steps):
            action_np, log_prob, _ = policy.get_action(obs)
            
            # FIX: Clamp the action sent to the environment, but store the 
            # raw action in the trajectory buffer.
            action_env = np.clip(action_np, -1.0, 1.0)
            next_obs, reward, done, truncated, info = env.step(action_env)

            traj["obs"].append(torch.tensor(obs, dtype=torch.float32, device=device))
            traj["actions"].append(torch.tensor(action_np, dtype=torch.float32, device=device))
            traj["log_probs"].append(
                log_prob if isinstance(log_prob, torch.Tensor) else torch.tensor(log_prob, device=device)
            )
            traj["rewards"].append(reward)
            traj["dones"].append(float(done or truncated))
            traj["total_return"] += reward

            obs = next_obs
            if done or truncated:
                break

        trajectories.append(traj)
    return trajectories


def grpo_update(policy: TransformerPolicyWrapper,
                value_fn: ValueFunction,
                policy_optimizer: torch.optim.Optimizer,
                trajectories_per_group: list,
                cfg: DictConfig,
                device: torch.device):
    policy.train()
    n_epochs  = int(getattr(cfg.training, "ppo_epochs", 4))
    clip_eps  = cfg.training.clip_eps
    total_policy_loss = 0.0
    total_return      = 0.0

    all_prepped = []
    for group in trajectories_per_group:
        returns  = [t["total_return"] for t in group]
        mean_ret = float(np.mean(returns))
        std_ret  = float(np.std(returns)) + 1e-8
        total_return += mean_ret

        for traj in group:
            if len(traj["obs"]) == 0:
                continue
            T         = len(traj["obs"])
            advantage = (traj["total_return"] - mean_ret) / std_ret
            all_prepped.append({
                "obs":           torch.stack(traj["obs"]),
                "actions":       torch.stack(traj["actions"]),
                "old_log_probs": torch.stack(traj["log_probs"]).detach(),
                "adv_t":         torch.full((T,), advantage, device=device, dtype=torch.float32),
            })

    for epoch in range(n_epochs):
        policy_optimizer.zero_grad()
        epoch_loss = 0.0

        for p in all_prepped:
            dist          = policy.forward(p["obs"])
            new_log_probs = dist.log_prob(p["actions"]).sum(-1)
            entropy       = dist.entropy().sum(-1).mean()

            ratio = (new_log_probs - p["old_log_probs"]).exp()
            surr  = torch.min(
                ratio * p["adv_t"],
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * p["adv_t"],
            )
            
            # FIX: Divide the loss by the number of prepped trajectories to 
            # prevent gradient explosion when .backward() accumulates them.
            loss = (-surr.mean() - cfg.training.entropy_coeff * entropy) / max(len(all_prepped), 1)
            loss.backward()
            
            # Re-multiply to log the correct un-scaled loss magnitude
            epoch_loss += loss.item() * max(len(all_prepped), 1)

        nn.utils.clip_grad_norm_(list(policy.parameters()), cfg.training.max_grad_norm)
        policy_optimizer.step()
        total_policy_loss += epoch_loss / max(len(all_prepped), 1)

    n_groups = max(len(trajectories_per_group), 1)
    return {
        "policy_loss": total_policy_loss / n_epochs,
        "mean_return": total_return / n_groups,
    }


# ---------------------------------------------------------------------------
# GRPO with world model (Part 2d)
# ---------------------------------------------------------------------------

class WorldModelWrapper:
    ACTION_MEAN = np.array([ 0.20769862830638885,  0.1172831580042839,  -0.1328626126050949,
                              0.006111515685915947,-0.0030502916779369116,-0.036297377198934555,
                              0.0], dtype=np.float32)
    ACTION_STD  = np.array([ 0.6475179195404053,   0.3413824141025543,   0.8427339792251587,
                              0.04650535434484482,  0.09098237007856369,  0.0969737097620964,
                              1.4071979522705078], dtype=np.float32)

    def __init__(self, checkpoint_path: str, device: torch.device):
        from omegaconf import OmegaConf
        sys.path.insert(0, os.path.dirname(__file__))
        from hw3.dreamerV3 import DreamerV3

        cfg = OmegaConf.create({
            "dataset": {"encode_with_t5": False, "chars_list": []},
            "max_block_size": 64,
            "device": str(device),
            "n_embd": 512,
        })

        self.model = DreamerV3(
            obs_shape=(3, 64, 64),
            action_dim=7,
            stoch_dim=32,
            discrete_dim=32,
            deter_dim=512,
            hidden_dim=512,
            cfg=cfg,
        ).to(device)
        self.device = device

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            model_sd = self.model.state_dict()
            matched   = {k: v for k, v in ckpt.items()
                         if k in model_sd and model_sd[k].shape == v.shape}
            unmatched = [k for k in ckpt if k not in model_sd or
                         model_sd[k].shape != ckpt[k].shape]
            print(f"[WorldModel] Loaded {len(matched)}/{len(ckpt)} keys from checkpoint")
            if unmatched:
                print(f"[WorldModel] Unmatched keys: {unmatched}")
            self.model.load_state_dict(matched, strict=False)
        else:
            print(f"[WorldModel] Non-dict checkpoint type: {type(ckpt)}")
        self.model.eval()
        self._rssm_state = self.model.get_initial_state(1, device)

    def reset_rssm_state(self):
        self._rssm_state = self.model.get_initial_state(1, self.device)

    def step(self, obs_13: np.ndarray, action_7: np.ndarray):
        norm_action = (action_7 - self.ACTION_MEAN) / (self.ACTION_STD + 1e-8)
        at = torch.tensor(norm_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            new_state, _ = self.model.rssm_step(self._rssm_state, at, embed=None)
            features = torch.cat([new_state['h'], new_state['z']], dim=-1)
            r_symlog = float(self.model.reward_head(features).squeeze())
            reward = float(np.sign(r_symlog) * (np.exp(abs(r_symlog)) - 1.0))

        self._rssm_state = new_state

        next_obs_13 = obs_13.copy().astype(np.float32)
        delta_pos = action_7[:3].astype(np.float32) * 0.01 
        next_obs_13[:3] += delta_pos
        next_obs_13[7:10]  -= delta_pos
        next_obs_13[10:13] -= delta_pos

        return next_obs_13, reward


def grpo_worldmodel_update(policy: TransformerPolicyWrapper,
                            world_model: WorldModelWrapper,
                            policy_optimizer: torch.optim.Optimizer,
                            current_obs: np.ndarray,
                            group_size: int,
                            horizon: int,
                            cfg: DictConfig,
                            device: torch.device):
    policy.train()
    policy_optimizer.zero_grad()

    group_trajs = []
    group_returns = []

    for _ in range(group_size):
        obs = current_obs.copy()
        traj_obs, traj_actions, traj_log_probs = [], [], []
        total_return = 0.0
        world_model.reset_rssm_state()

        for _ in range(horizon):
            action_np, log_prob, _ = policy.get_action(obs)
            
            # FIX: Clamp before stepping the world model, save raw.
            action_env = np.clip(action_np, -1.0, 1.0)
            next_obs, reward = world_model.step(obs, action_env)

            traj_obs.append(torch.tensor(obs, dtype=torch.float32, device=device))
            traj_actions.append(torch.tensor(action_np, dtype=torch.float32, device=device))
            traj_log_probs.append(
                log_prob if isinstance(log_prob, torch.Tensor)
                else torch.tensor(log_prob, device=device)
            )
            total_return += reward
            obs = next_obs

        group_trajs.append({
            "obs": traj_obs, "actions": traj_actions,
            "log_probs": traj_log_probs, "total_return": total_return,
        })
        group_returns.append(total_return)

    mean_ret = float(np.mean(group_returns))
    std_ret  = float(np.std(group_returns)) + 1e-8

    prepped = []
    for traj in group_trajs:
        if len(traj["obs"]) == 0:
            continue
        T = len(traj["obs"])
        advantage = (traj["total_return"] - mean_ret) / std_ret
        prepped.append({
            "obs":          torch.stack(traj["obs"]),
            "actions":      torch.stack(traj["actions"]),
            "old_log_probs": torch.stack(traj["log_probs"]).detach(),
            "adv_t":        torch.full((T,), advantage, device=device, dtype=torch.float32),
        })

    n_epochs = int(getattr(cfg.training, "ppo_epochs", 4))
    clip_eps = cfg.training.clip_eps
    total_policy_loss = 0.0

    for epoch in range(n_epochs):
        policy_optimizer.zero_grad()
        epoch_loss = 0.0

        for p in prepped:
            dist          = policy.forward(p["obs"])
            new_log_probs = dist.log_prob(p["actions"]).sum(-1)
            entropy       = dist.entropy().sum(-1).mean()

            ratio = (new_log_probs - p["old_log_probs"]).exp()
            surr  = torch.min(
                ratio * p["adv_t"],
                ratio.clamp(1 - clip_eps, 1 + clip_eps) * p["adv_t"],
            )
            
            # FIX: Gradient accumulation division.
            loss = (-surr.mean() - cfg.training.entropy_coeff * entropy) / max(len(prepped), 1)
            loss.backward()
            
            epoch_loss += loss.item() * max(len(prepped), 1)

        nn.utils.clip_grad_norm_(list(policy.parameters()), cfg.training.max_grad_norm)
        policy_optimizer.step()
        total_policy_loss += epoch_loss / max(len(prepped), 1)

    return {
        "policy_loss": total_policy_loss / n_epochs,
        "mean_imagined_return": mean_ret,
        "imagined_return_std": std_ret,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@hydra.main(config_path="conf", config_name="transformer_rl", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.r_seed)
    np.random.seed(cfg.r_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.experiment.project,
        name=cfg.experiment.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    task_id = int(cfg.sim.eval_tasks[0])
    env = FastLIBEROEnv(task_id=task_id, max_episode_steps=cfg.sim.episode_length, cfg=cfg)

    obs_dim = env.obs_dim
    action_dim = env._action_dim

    policy = TransformerPolicyWrapper(cfg.init_checkpoint, device, cfg, obs_dim=obs_dim)
    value_fn = ValueFunction(obs_dim, cfg.value.hidden_dim, cfg.value.n_layers).to(device)

    # FIX: Use parameter groups to protect pre-trained weights from the "cold start" adapter.
    if policy.pose_adapter is not None:
        adapter_params = list(policy.pose_adapter.parameters())
        transformer_params = list(policy.model.parameters())
        policy_optimizer = torch.optim.Adam([
            {'params': transformer_params, 'lr': 1e-5}, # Tiny LR for pre-trained weights
            {'params': adapter_params, 'lr': cfg.training.learning_rate} # Normal LR for new adapter
        ])
    else:
        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)

    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=cfg.value.learning_rate)

    algorithm = cfg.rl.algorithm.lower()

    if algorithm == "ppo":
        buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)

        obs, _ = env.reset()
        policy.reset_context()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        total_steps = 0
        episode_returns, episode_successes = [], []
        ep_ret = 0.0

        while total_steps < cfg.training.total_env_steps:
            buffer.reset()
            with torch.no_grad():
                for _ in range(cfg.training.rollout_length):
                    action_np, log_prob, _ = policy.get_action(obs)
                    value = value_fn(obs_t.unsqueeze(0))
                    
                    # FIX: Clamp for env, store raw
                    action_env = np.clip(action_np, -1.0, 1.0)
                    next_obs, reward, done, truncated, info = env.step(action_env)
                    
                    ep_ret += reward
                    total_steps += 1
                    buffer.add(
                        obs_t,
                        torch.tensor(action_np, device=device),
                        log_prob if isinstance(log_prob, torch.Tensor) else torch.tensor(log_prob, device=device),
                        torch.tensor(reward, device=device),
                        value.squeeze(0),
                        torch.tensor(float(done or truncated), device=device),
                    )
                    if done or truncated:
                        episode_returns.append(ep_ret)
                        episode_successes.append(float(info.get("success_placed", 0.0)))
                        ep_ret = 0.0
                        obs, _ = env.reset()
                        policy.reset_context()
                    else:
                        obs = next_obs
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                    if buffer.full():
                        break
                last_value = value_fn(obs_t.unsqueeze(0)).squeeze(0)

            returns, advantages = buffer.compute_returns_and_advantages(
                last_value, cfg.training.gamma, cfg.training.gae_lambda
            )
            update_info = ppo_update(policy, value_fn, policy_optimizer, value_optimizer, buffer, returns, advantages, cfg)

            if total_steps % cfg.log_interval < cfg.training.rollout_length:
                log = {"train/total_steps": total_steps, **{f"train/{k}": v for k, v in update_info.items()}}
                if episode_returns:
                    log["train/episode_return"] = np.mean(episode_returns[-10:])
                    log["train/success_rate"] = np.mean(episode_successes[-10:])
                wandb.log(log, step=total_steps)
                print(f"[PPO {total_steps}] return={log.get('train/episode_return', float('nan')):.3f}")

    elif algorithm == "grpo":
        total_steps = 0
        update_count = 0
        all_returns = []

        while total_steps < cfg.training.total_env_steps:
            trajectories_per_group = []
            for _ in range(cfg.grpo.num_groups):
                env.reset()
                sim = env.env.sim
                init_state = sim.get_state()
                group = collect_grpo_group(
                    env, policy, init_state,
                    cfg.grpo.group_size, cfg.sim.episode_length, device,
                )
                trajectories_per_group.append(group)
                total_steps += sum(len(t["obs"]) for t in group)

            update_info = grpo_update(policy, value_fn, policy_optimizer,
                                      trajectories_per_group, cfg, device)
            update_count += 1
            all_returns.extend([t["total_return"] for g in trajectories_per_group for t in g])

            log = {
                "train/total_steps": total_steps,
                "train/update": update_count,
                **{f"train/{k}": v for k, v in update_info.items()},
                "train/episode_return": np.mean(all_returns[-50:]) if all_returns else 0.0,
            }
            wandb.log(log, step=total_steps)
            print(f"[GRPO {total_steps}] return={log['train/episode_return']:.3f} "
                  f"policy_loss={update_info['policy_loss']:.4f}")

    elif algorithm == "grpo_worldmodel":
        wm_checkpoint = cfg.get("world_model_checkpoint", None)
        if wm_checkpoint is None:
            raise ValueError("grpo_worldmodel requires world_model_checkpoint in config")

        world_model = WorldModelWrapper(wm_checkpoint, device)

        horizon     = int(cfg.grpo.get("horizon", 50))
        group_size  = int(cfg.grpo.group_size)
        total_steps = 0
        update_count = 0
        all_imagined_returns = []

        obs, _ = env.reset()
        ep_steps = 0

        print(f"[GRPO-WM] horizon={horizon}, group_size={group_size}")

        while total_steps < cfg.training.total_env_steps:
            update_info = grpo_worldmodel_update(
                policy, world_model, policy_optimizer,
                obs, group_size, horizon, cfg, device,
            )
            total_steps += group_size * horizon
            update_count += 1
            all_imagined_returns.append(update_info["mean_imagined_return"])

            with torch.no_grad():
                action_np, _, _ = policy.get_action(obs)
                
            # FIX: Clamp for the real step.
            action_env = np.clip(action_np, -1.0, 1.0)
            obs, _, done, truncated, _ = env.step(action_env)
            
            ep_steps += 1
            if done or truncated:
                obs, _ = env.reset()
                ep_steps = 0

            if update_count % max(1, cfg.log_interval // (group_size * horizon)) == 0:
                log = {
                    "train/total_steps": total_steps,
                    "train/update": update_count,
                    **{f"train/{k}": v for k, v in update_info.items()},
                    "train/imagined_return": np.mean(all_imagined_returns[-20:]),
                }
                wandb.log(log, step=total_steps)
                print(f"[GRPO-WM {total_steps}] "
                      f"imagined_return={log['train/imagined_return']:.3f}  "
                      f"policy_loss={update_info['policy_loss']:.4f}  "
                      f"return_std={update_info['imagined_return_std']:.4f}")

    else:
        raise ValueError(f"Unknown rl.algorithm: {algorithm}. Choose 'ppo', 'grpo', or 'grpo_worldmodel'.")

    torch.save({
        "policy": {k: v for k, v in policy.model.state_dict().items()},
        "value_fn": value_fn.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, "transformer_rl_final.pth")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()