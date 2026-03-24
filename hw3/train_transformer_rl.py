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

    The transformer policy expects a history of observations and actions;
    this wrapper maintains the required context window internally.
    """

    def __init__(self, checkpoint_path: str, device: torch.device, cfg: DictConfig):
        import dill
        # The HW1 checkpoint is saved as a full model object via torch.save(model, path, pickle_module=dill)
        self.model = torch.load(checkpoint_path, map_location=device, pickle_module=dill)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.cfg = cfg
        self._context = []
        # Register a learnable log_std on the model itself so it appears in model.parameters()
        # and is saved/loaded with the checkpoint naturally.
        action_dim = cfg.policy.action_dim
        self.model.rl_log_std = nn.Parameter(torch.zeros(action_dim, device=device))

    def reset_context(self):
        self._context = []

    def _compute_action_mean(self, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Run the GRP model forward on a batch of state vectors and return action means.

        The GRP model can operate in two modes depending on the checkpoint:
          - Newer API: encode_pose() + forward(observations=None, ..., pose=pose, ...)
          - Legacy API: dummy image/goal tensors + forward(images, goals, goal_imgs, pose=pose)
        """
        B = obs_t.shape[0]
        model_cfg = self.model._cfg
        action_dim = self.cfg.policy.action_dim

        if hasattr(self.model, 'encode_pose'):
            # Newer GRP API: privileged-state-only mode (no images needed)
            pose = self.model.encode_pose(obs_t.unsqueeze(1))  # (B, 1, obs_dim) -> encoded
            out = self.model.forward(
                observations=None, text_goal=None, goal_image=None,
                mask_=True, pose=pose, last_action=None,
            )
            raw = out['actions'] if isinstance(out, dict) else (out[0] if isinstance(out, tuple) else out)
        else:
            # Legacy API: feed zeros for image/goal inputs, state as pose
            h = model_cfg.image_shape[0]
            w = model_cfg.image_shape[1]
            stacking = model_cfg.policy.obs_stacking
            dummy_img = torch.zeros(B, h, w, 3 * stacking, device=self.device)
            dummy_goal_txt = torch.zeros(B, model_cfg.max_block_size, dtype=torch.long, device=self.device)
            dummy_goal_img = torch.zeros(B, h, w, 3, device=self.device)
            encoded_pose = self.model.encode_state(obs_t.unsqueeze(1))
            out, _ = self.model.forward(dummy_img, dummy_goal_txt, dummy_goal_img, pose=encoded_pose)
            raw = out

        decoded = self.model.decode_action(raw)  # undo action normalisation
        # The model may predict action_stacking steps; take only the first action_dim values
        return decoded.reshape(B, -1)[:, :action_dim]

    def forward(self, obs: torch.Tensor):
        """
        Return a Normal distribution over actions for a batched obs tensor.
        Called by ppo_update() during the gradient update step.
        """
        mean = self._compute_action_mean(obs)
        std = self.model.rl_log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """
        Query the transformer for an action given the current observation.

        Args:
            obs: (obs_dim,) numpy array
            deterministic: if True return mean, else sample
        Returns:
            action: (action_dim,) numpy array
            log_prob: scalar tensor
            entropy: scalar tensor
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)

        dist = self.forward(obs_t)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        action = action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return (
            action.squeeze(0).cpu().detach().numpy(),
            log_prob.squeeze(0).detach(),
            entropy.squeeze(0).detach(),
        )

    def parameters(self):
        return self.model.parameters()

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
    """
    Reset to the same initial state and collect `group_size` trajectories.

    Returns a list of trajectory dicts, each containing:
        obs, actions, log_probs, rewards, dones, total_return
    """
    trajectories = []
    for _ in range(group_size):
        obs, info = env.reset(options={"init_state": init_state})
        policy.reset_context()

        traj = {
            "obs": [], "actions": [], "log_probs": [],
            "rewards": [], "dones": [], "total_return": 0.0,
        }

        for _ in range(max_steps):
            action_np, log_prob, _ = policy.get_action(obs)
            next_obs, reward, done, truncated, info = env.step(action_np)

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
    """
    GRPO update: compute group-relative advantages and update policy.

    Args:
        trajectories_per_group: list of lists; each inner list is a group of
            trajectory dicts collected from the same initial state.
    Returns:
        dict with "policy_loss", "mean_return"
    """
    policy.train()
    total_policy_loss = 0.0
    total_return = 0.0
    n_updates = 0

    for group in trajectories_per_group:
        # --- Group-relative advantage normalisation ---
        # Each trajectory gets a scalar advantage: how much better/worse it did
        # than the group average, normalised by the group's standard deviation.
        returns = [t["total_return"] for t in group]
        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns)) + 1e-8
        total_return += mean_ret

        for traj in group:
            if len(traj["obs"]) == 0:
                continue

            obs_t = torch.stack(traj["obs"])                          # (T, obs_dim)
            actions_t = torch.stack(traj["actions"])                  # (T, action_dim)
            old_log_probs_t = torch.stack(traj["log_probs"]).detach() # (T,)
            T = obs_t.shape[0]

            # Scalar advantage broadcast to every step in the trajectory
            advantage = (traj["total_return"] - mean_ret) / std_ret
            adv_t = torch.full((T,), advantage, device=device, dtype=torch.float32)

            # Re-evaluate the policy under its current parameters
            dist = policy.forward(obs_t)
            new_log_probs = dist.log_prob(actions_t).sum(-1)  # (T,)
            entropy = dist.entropy().sum(-1).mean()

            # Clipped surrogate objective (same as PPO)
            ratio = (new_log_probs - old_log_probs_t).exp()
            clip_eps = cfg.training.clip_eps
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean() - cfg.training.ent_coef * entropy

            policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()), cfg.training.max_grad_norm)
            policy_optimizer.step()

            total_policy_loss += policy_loss.item()
            n_updates += 1

    n_groups = max(len(trajectories_per_group), 1)
    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "mean_return": total_return / n_groups,
    }


# ---------------------------------------------------------------------------
# GRPO with world model (Part 2d)
# ---------------------------------------------------------------------------

def grpo_worldmodel_update(policy: TransformerPolicyWrapper,
                            world_model,
                            current_obs: np.ndarray,
                            group_size: int,
                            horizon: int,
                            cfg: DictConfig,
                            device: torch.device):
    """
    GRPO using the HW2 world model to generate imagined trajectories.

    Args:
        world_model: trained HW2 world model (SimpleWorldModel or DreamerV3)
        current_obs: (obs_dim,) current real observation used as rollout start
        group_size: number of imagined trajectories per state
        horizon: number of imagination steps
    Returns:
        dict with "policy_loss", "mean_imagined_return"
    """
    policy.train()
    obs_t = torch.tensor(current_obs, dtype=torch.float32, device=device).unsqueeze(0)  # (1, obs_dim)

    group_trajs = []
    group_returns = []

    # ------------------------------------------------------------------
    # Collect group_size imagined trajectories from the same start state
    # ------------------------------------------------------------------
    for _ in range(group_size):
        pose = obs_t.clone()  # (1, obs_dim) — current (real) observation as start
        # Encode into the world model's normalised state space
        encoded_pose = world_model.encode_state(pose) if hasattr(world_model, 'encode_state') else pose

        traj_obs, traj_actions, traj_log_probs, traj_rewards = [], [], [], []
        total_return = 0.0

        for _ in range(horizon):
            obs_np = pose.squeeze(0).cpu().detach().numpy()
            action_np, log_prob, _ = policy.get_action(obs_np)
            action_t = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

            # Encode action for the world model and step forward
            encoded_act = world_model.encode_action(action_t) if hasattr(world_model, 'encode_action') else action_t
            next_pose_enc, reward_pred = world_model.forward(encoded_pose, encoded_act)

            # Decode back to original observation space for the policy
            next_pose = world_model.decode_state(next_pose_enc) if hasattr(world_model, 'decode_state') else next_pose_enc

            r = reward_pred.squeeze().item()
            total_return += r

            traj_obs.append(pose.squeeze(0))
            traj_actions.append(action_t.squeeze(0))
            traj_log_probs.append(
                log_prob if isinstance(log_prob, torch.Tensor) else torch.tensor(log_prob, device=device)
            )
            traj_rewards.append(r)

            pose = next_pose.detach()
            encoded_pose = next_pose_enc.detach()

        group_trajs.append({
            "obs": traj_obs,
            "actions": traj_actions,
            "log_probs": traj_log_probs,
            "total_return": total_return,
        })
        group_returns.append(total_return)

    # ------------------------------------------------------------------
    # GRPO update: group-relative advantages + clipped surrogate loss
    # ------------------------------------------------------------------
    mean_ret = float(np.mean(group_returns))
    std_ret = float(np.std(group_returns)) + 1e-8
    total_policy_loss = 0.0
    n_updates = 0

    for traj in group_trajs:
        T = len(traj["obs"])
        if T == 0:
            continue

        obs_stack = torch.stack(traj["obs"])                           # (T, obs_dim)
        actions_stack = torch.stack(traj["actions"])                   # (T, action_dim)
        old_log_probs = torch.stack(traj["log_probs"]).detach()        # (T,)

        advantage = (traj["total_return"] - mean_ret) / std_ret
        adv_t = torch.full((T,), advantage, device=device, dtype=torch.float32)

        dist = policy.forward(obs_stack)
        new_log_probs = dist.log_prob(actions_stack).sum(-1)

        ratio = (new_log_probs - old_log_probs).exp()
        clip_eps = cfg.training.clip_eps
        surr = torch.min(
            ratio * adv_t,
            ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t,
        )
        loss = -surr.mean()
        loss.backward()  # accumulate; caller is responsible for optimizer.zero_grad / .step

        total_policy_loss += loss.item()
        n_updates += 1

    # Clip accumulated gradients
    nn.utils.clip_grad_norm_(list(policy.parameters()), cfg.training.max_grad_norm)

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "mean_imagined_return": mean_ret,
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

    # Load transformer policy from HW1 checkpoint
    policy = TransformerPolicyWrapper(cfg.init_checkpoint, device, cfg)

    # Separate value function (required by hw3.md)
    value_fn = ValueFunction(obs_dim, cfg.value.hidden_dim, cfg.value.n_layers).to(device)

    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.learning_rate)
    value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=cfg.value.learning_rate)

    algorithm = cfg.rl.algorithm.lower()

    if algorithm == "ppo":
        # ------------------------------------------------------------------
        # PPO loop (reuses the buffer + update from Part 1)
        # ------------------------------------------------------------------
        buffer = RolloutBuffer(cfg.training.rollout_length, obs_dim, action_dim, device)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_fn.parameters()),
            lr=cfg.training.learning_rate,
        )

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
                    next_obs, reward, done, truncated, info = env.step(action_np)
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
            update_info = ppo_update(policy, value_fn, optimizer, buffer, returns, advantages, cfg)

            if total_steps % cfg.log_interval < cfg.training.rollout_length:
                log = {"train/total_steps": total_steps, **{f"train/{k}": v for k, v in update_info.items()}}
                if episode_returns:
                    log["train/episode_return"] = np.mean(episode_returns[-10:])
                    log["train/success_rate"] = np.mean(episode_successes[-10:])
                wandb.log(log, step=total_steps)
                print(f"[PPO {total_steps}] return={log.get('train/episode_return', float('nan')):.3f}")

    elif algorithm == "grpo":
        # ------------------------------------------------------------------
        # GRPO loop with ground-truth resets (Part 2c)
        # ------------------------------------------------------------------
        total_steps = 0
        update_count = 0
        all_returns = []

        while total_steps < cfg.training.total_env_steps:
            # Collect groups: reset to different initial states
            # For each group we: (1) reset the env to get a fresh random state,
            # (2) snapshot that state, (3) re-run the policy group_size times from
            # the exact same snapshot so GRPO can compare within-group outcomes.
            trajectories_per_group = []
            for _ in range(cfg.rl.num_groups):
                env.reset()  # randomise initial state
                init_state = env.env.sim.get_state()  # MuJoCo SimState snapshot
                group = collect_grpo_group(
                    env, policy, init_state,
                    cfg.rl.group_size, cfg.sim.episode_length, device,
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

    else:
        raise ValueError(f"Unknown rl.algorithm: {algorithm}. Choose 'ppo' or 'grpo'.")

    # Save final checkpoint
    torch.save({
        "policy": {k: v for k, v in policy.model.state_dict().items()},
        "value_fn": value_fn.state_dict(),
        "cfg": OmegaConf.to_container(cfg),
    }, "transformer_rl_final.pth")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
