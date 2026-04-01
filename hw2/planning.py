

from dreamerV3 import GRPBase
import torch


class Planner(GRPBase):
    """
    Base class for planners. Defines the interface for planning algorithms.
    """
    def __init__(self, cfg=None):
        super(Planner, self).__init__(cfg)

    def update(self, states, actions):
        """
        Update the planner's internal model or policy based on collected states and actions.
        This method can be overridden by planners that learn from data (e.g., PolicyPlanner).
        
        Args:
            states: Tensor of shape (B, state_dim) containing collected states
            actions: Tensor of shape (B, action_dim) containing collected actions
        """
        pass  # Default implementation does nothing
    
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences given an initial state.
        
        Args:
            initial_state: Dictionary containing initial state information
            return_best_sequence: If True, returns the best action sequence; else returns action mean
            
        Returns:
            actions: Tensor of shape (horizon, action_dim) with the planned action sequence
            predicted_reward: Float value of the expected cumulative reward for the planned sequence
        """
        raise NotImplementedError("Plan method must be implemented by subclasses")

class CEMPlanner(Planner):
    """
    Cross-Entropy Method (CEM) planner for model-based planning.
    Samples action sequences and uses a world model to find high-reward plans.
    """
    def __init__(self, 
                 world_model,
                 action_dim,
                 cfg):
        """
        Initialize CEM planner.
        
        Args:
            world_model: World model (DreamerV3 or SimpleWorldModel) used for imagining future trajectories
            action_dim: Dimensionality of the action space
            cfg: Configuration object
        """
        # TODO: Part 1.3 - Initialize CEM planner
        ## Set up world model reference and determine if using DreamerV3 or SimpleWorldModel
        super().__init__(cfg)
        self.world_model = world_model
        self.action_dim = action_dim
        self.cfg = cfg

        from simple_world_model import SimpleWorldModel
        self.is_simple_model = isinstance(world_model, SimpleWorldModel)

        self.horizon = cfg.planner.horizon
        self.iterations = cfg.planner.num_iterations
        self.num_samples = cfg.planner.num_samples
        self.num_elites = cfg.planner.num_elites
        self.temperature = cfg.planner.temperature
        
        
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences using CEM to maximize predicted rewards.
        
        Args:
            initial_state: Dictionary containing initial state 
                          - For DreamerV3: {'h', 'z', 'z_probs'}
                          - For SimpleWorldModel: {'pose'}
            return_best_sequence: If True, returns best action sequence; else returns action mean
            
        Returns:
            best_actions: Tensor of shape (horizon, action_dim) with the best action sequence
            best_reward: Float value of the sum of predicted rewards for the best sequence
        """
        # TODO: Part 1.3 - Implement CEM planning algorithm
        ## Sample action sequences, evaluate with world model, select elites, update distribution
        device = next(self.world_model.parameters()).device

        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones(self.horizon, self.action_dim, device=device)

        with torch.no_grad():
            for i in range(self.iterations):
                samples = mean + std * torch.randn(self.num_samples, self.horizon, self.action_dim, device=device)
                samples = torch.clamp(samples, -1.0, 1.0)
                
                returns = self._evaluate_sequences(initial_state, samples)

                _, elite_indices = torch.topk(returns, self.num_elites)
                elites = samples[elite_indices]

                mean = torch.mean(elites, dim=0)
                std = torch.std(elites, dim=0) + 1e-6

            if return_best_sequence:
                best_idx = torch.argmax(returns)
                return samples[best_idx], torch.max(returns)
            
            return mean, torch.mean(returns)
    
    def _evaluate_sequences(self, initial_state, action_sequences):
        """
        Evaluate a batch of action sequences by rolling them out in the world model.
        
        Args:
            initial_state: Dictionary with initial state (RSSM state for DreamerV3 or pose for SimpleWorldModel)
            action_sequences: Tensor of shape (num_samples, horizon, action_dim)
            
        Returns:
            rewards: Tensor of shape (num_samples,) with sum of predicted rewards
        """
        # TODO: Part 1.3 - Route to appropriate evaluation method
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        # 1. Route to SimpleWorldModel evaluation
        if self.is_simple_model:
            return self._evaluate_sequences_simple(initial_state, action_sequences)
        
        # 2. Route to DreamerV3 evaluation
        else:
            return self._evaluate_sequences_dreamer(initial_state, action_sequences)
        pass
    
    def _evaluate_sequences_dreamer(self, initial_state, action_sequences):
        """
        Evaluate sequences using DreamerV3 RSSM-based rollout.
        """
        # TODO: Part 3.3 - Implement CEM planning with DreamerV3
        ## Roll out action sequences in the DreamerV3 world model and compute total rewards
        num_samples, horizon, action_dim = action_sequences.shape
        device = action_sequences.device
        
        current_state = {}
        for key, value in initial_state.items():
            if value is not None:
                current_state[key] = value.repeat(num_samples, *([1] * (value.dim() - 1)))
                
        total_rewards = torch.zeros(num_samples, 1, device=device)
        
        self.world_model.eval()
        with torch.no_grad():
            for t in range(horizon):
                actions = action_sequences[:, t, :]
                
                _, next_state = self.world_model.rssm_step(current_state, actions, embed=None)
                
                features = torch.cat([next_state['h'], next_state['z']], dim=-1)
                
                reward = self.world_model.reward_head(features)
                
                total_rewards += reward
                
                current_state = next_state
                
        return total_rewards.squeeze(-1)
        pass
    
    def _evaluate_sequences_simple(self, initial_state, action_sequences):
        """
        Evaluate sequences using SimpleWorldModel pose-based rollout.
        """
        # TODO: Part 1.3 - Implement CEM planning with SimpleWorldModel
        ## Roll out action sequences using SimpleWorldModel and compute total rewards
        num_samples = action_sequences.shape[0]
        current_pose = initial_state["pose"].to(self.cfg.device).float()

        if current_pose.dim() == 1:
            current_pose = current_pose.unsqueeze(0)
        elif current_pose.dim() == 3:
            current_pose = current_pose.squeeze(1)

        current_pose = current_pose.repeat(num_samples, 1).contiguous()
        total_rewards = torch.zeros(num_samples, device=self.cfg.device)

        self.world_model.eval()
        
        
        for t in range(self.horizon):
            actions_t = action_sequences[:, t, :]
            next_pose_pred, reward_pred = self.world_model(current_pose, actions_t)
            total_rewards += reward_pred.view(-1)
            current_pose = next_pose_pred
            
        return total_rewards
        
    
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that works with both DreamerV3 and SimpleWorldModel.
        This wrapper obtains the current state and plans actions.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (for DreamerV3)
            prev_actions: Previous actions (optional, for state initialization)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Planned action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': Expected cumulative reward
                - 'final_state': Final state after processing inputs
        """
        # TODO: Part 1.3 - Route forward pass to appropriate model
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if self.is_simple_model:
            # Handle edge case where eval script passes pose positionally into observations
            if pose is None and observations is not None:
                pose = observations
            return self._forward_simple(pose, return_full_sequence)
            
        # 2. Route to DreamerV3
        else:
            return self._forward_dreamer(observations, prev_actions, prev_state, return_full_sequence)
    
    def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
        """Forward pass for DreamerV3 model."""
        # TODO: Part 4.2 - Implement DreamerV3 forward pass for policy
        ## Encode observations, roll through RSSM, and plan with policy from current state
        B = observations.shape[0]
        device = observations.device
        
        latest_obs = observations[:, -1]
        
        embed = self.world_model.encoder(latest_obs)
        
        if prev_state is None:
            prev_state = self.world_model.get_initial_state(B, device)
        if prev_actions is None:
            last_action = torch.zeros(B, self.action_dim, device=device)
        else:
            last_action = prev_actions[:, -1]
            
        current_state, _ = self.world_model.rssm_step(prev_state, last_action, embed)
        
        actions_seq, expected_reward = self.plan(current_state)

        if return_full_sequence:
            actions_out = actions_seq.unsqueeze(0) 
        else:
            actions_out = actions_seq[0].unsqueeze(0) 
            
        return {
            "actions": actions_out,
            "predicted_rewards": expected_reward,
            "final_state": current_state
        }

    def _forward_simple(self, pose, return_full_sequence=False):
        """Forward pass for SimpleWorldModel inside CEM Planner."""
        current_state = {"pose": pose}
        
        # CEM plan() returns both the sequence and the expected reward
        actions_seq, expected_reward = self.plan(current_state)
        
        if return_full_sequence:
            actions_out = actions_seq.unsqueeze(0)
        else:
            # Extract just the first action for execution and keep batch dim
            actions_out = actions_seq[0].unsqueeze(0)
            
        return {
            "actions": actions_out,
            "predicted_rewards": expected_reward,
            "final_state": current_state
        }

        # [Imagine method remains mostly the same, ensuring valid input shapes for heads]
    def preprocess_state(self, image):
        """Preprocess observation image"""
        # TODO: Preprocess image for input
        ## Resize, normalize, and convert to channel-first format
        return self.world_model.preprocess_state(image)
        pass


class PolicyPlanner(GRPBase):
    """
    Policy-based planner that uses a trained policy model to generate action sequences.
    Rolls out the policy over a horizon by predicting actions and states at each timestep.
    """
    def __init__(self, 
                 world_model,
                 policy_model,
                 action_dim,
                 cfg=None,
                 horizon=None):
        """
        Initialize Policy planner.
        
        Args:
            world_model: World model (DreamerV3 or SimpleWorldModel) used for predicting future states
            policy_model: Trained policy model that predicts actions given states
            action_dim: Dimensionality of the action space
            cfg: Configuration object
            horizon: Planning horizon (number of timesteps to plan ahead)
        """
        # TODO: Part 2.2 - Initialize Policy planner
        ## Set up world model, policy model, optimizer, and scheduler
        self.world_model = world_model
        self.policy_model = policy_model
        self.action_dim = action_dim
        self.cfg = cfg
        self.batch_size = self.cfg.batch_size
        self.epochs = self.cfg.max_iters

        self.horizon = horizon if horizon is not None else (cfg.horizon if cfg else 10)
        lr = cfg.policy if cfg and hasattr(cfg, "policy_lr") else 1e-3
        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.9)

        self.criterion = nn.MSELoss()

    def update(self, states, actions):
        """
        Docstring for update
        Update the policy model using collected states and actions.
        
        :param self: Description
        :param states: Description
        :param actions: Description
        """
        # TODO: Part 2.2 - Implement policy training
    
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)


        state_mean = states.mean(dim=0, keepdim=True)
        state_std = states.std(dim=0, keepdim=True) + 1e-6 # Epsilon to avoid division by zero
        normalized_states = (states - state_mean) / state_std


        batch_size = self.cfg.batch_size if self.cfg and hasattr(self.cfg, 'batch_size') else 256
        
        from torch.utils.data import TensorDataset, DataLoader, random_split
        dataset = TensorDataset(normalized_states, actions)
        

        val_size = max(1, int(len(dataset) * 0.1)) 
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


        self.policy_model.train()
        total_train_loss = 0.0
        

        for batch_states, batch_actions in train_loader:
            self.optimizer.zero_grad()
            

            predicted_actions = self.policy_model(batch_states)
            loss = self.criterion(predicted_actions, batch_actions)
            
            loss.backward()
            self.optimizer.step()
            total_train_loss += loss.item()
            

        if hasattr(self, 'scheduler'):
            self.scheduler.step()


        self.policy_model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_states, val_actions in val_loader:
                val_preds = self.policy_model(val_states)
                val_loss = self.criterion(val_preds, val_actions)
                total_val_loss += val_loss.item()
                

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        

        print(f"Policy Update | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    
    def plan(self, initial_state, return_best_sequence=True):
        """
        Plan action sequences by rolling out the policy model over the horizon.
        
        Args:
            initial_state: Dictionary containing initial state 
                          - For DreamerV3: {'h', 'z', 'z_probs'}
                          - For SimpleWorldModel: {'pose'}
            return_best_sequence: If True, returns the planned sequence (unused here for consistency)
            
        Returns:
            actions: Tensor of shape (horizon, action_dim) with the planned action sequence
            total_reward: Float value of the sum of predicted rewards
        """
        # TODO: Part 2.2 - Implement policy rollout planning
        ## Roll out the policy over the horizon, predicting actions and accumulating rewards
        self.policy_model.eval()

        actions_seq = []
        total_reward = 0.0
        current_state = initial_state
        is_simple_model = "pose" in current_state

        with torch.no_grad():
            for t in range(self.horizon):
                if is_simple_model:
                    state_tensor = current_state["pose"]
                    action = self.policy_model(state_tensor)
                    actions_seq.append(action)
                    next_state, reward = self.world_model.step(current_state, action)
                else:
                    state_tensor = torch.cat([current_state["h"], current_state["z"]], dim=-1)
                    action = self.policy_model(state_tensor)
                    actions_seq.append(action)
                    
                    _, next_state = self.world_model.rssm_step(current_state, action, embed=None)
                    
                    features = torch.cat([next_state['h'], next_state['z']], dim=-1)
                    reward = self.world_model.reward_head(features)

                total_reward += reward
                current_state = next_state

        actions = torch.stack(actions_seq, dim=1)

        return actions, total_reward
    
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that works with both DreamerV3 and SimpleWorldModel.
        This wrapper obtains the current state and plans actions using the policy.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (for DreamerV3)
            prev_actions: Previous actions (optional, for state initialization)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Planned action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': Expected cumulative reward
                - 'final_state': Final state after processing inputs
        """
        # TODO: Part 2.2 - Route forward pass to appropriate model
        ## Determine if using DreamerV3 or SimpleWorldModel and call appropriate method
        if observations is not None:
            return self._forward_dreamer(
                observation=observations,
                prev_actions=prev_actions,
                prev_state=prev_state,
                return_full_sequence=return_full_sequence
            )
        
        elif pose is not None:
            return self._forward_simple(
                pose=pose,
                return_full_sequence=return_full_sequence
            )

        else:
            raise ValueError("Invalid input: Must provide either 'observations' for DreamerV3 or 'pose' for SimpleWorldModel.")
    
    def _forward_dreamer(self, observations, prev_actions, prev_state, return_full_sequence):
        """Forward pass for DreamerV3 model."""
        # TODO: Part 4.2 - Implement DreamerV3 forward pass for policy
        ## Encode observations, roll through RSSM, and plan with policy from current state
        B = observations.shape[0]
        device = observations.device
        latest_obs = observations[:, -1]
        
        self.world_model.eval() 
        
        with torch.no_grad(): 
            embed = self.world_model.encoder(latest_obs)
            
            if prev_state is None:
                prev_state = self.world_model.get_initial_state(B, device)
            if prev_actions is None:
                last_action = torch.zeros(B, self.action_dim, device=device)
            else:
                last_action = prev_actions[:, -1]
                
            current_state, _ = self.world_model.rssm_step(prev_state, last_action, embed)
            actions_seq, expected_reward = self.plan(current_state)
        
        if return_full_sequence:
            actions_out = actions_seq
        else:
            actions_out = actions_seq[:, 0, :]
            
        return {
            "actions": actions_out,
            "predicted_rewards": expected_reward,
            "final_state": current_state
        }
        
    
    def _forward_simple(self, pose, return_full_sequence):
        """Forward pass for SimpleWorldModel."""
        # TODO: Part 2.2 - Implement SimpleWorldModel forward pass for policy
        ## Plan from current pose using policy with SimpleWorldModel
        current_state = {"pose": pose}

        actions_seq, expected_reward = self.plan(current_state)

        if return_full_sequence:
            actions_out = actions_seq
        else:
            actions_out = actions_seq[:, 0, :]

        return {
            "actions": actions_out,
            "predicted_rewards": expected_reward,
            "final_state": current_state
        }
        pass


class RandomPlanner(GRPBase):
    """
    Random action planner that generates random actions uniformly distributed between -1 and 1.
    Useful as a baseline for comparing planning algorithms.
    """
    def __init__(self, 
                 action_dim,
                 cfg):
        """
        Initialize Random planner.
        
        Args:
            world_model: World model (optional, not used but kept for API compatibility)
            action_dim: Dimensionality of the action space (default: 7)
            cfg: Configuration object (optional)
            horizon: Planning horizon (number of timesteps to plan ahead)
        """
        super(RandomPlanner, self).__init__(cfg)
        
        self.action_dim = action_dim
            
    def forward(self, observations=None, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None, return_full_sequence=False):
        """
        Unified interface for planning that generates random actions.
        
        Args:
            observations: Tensor of shape (B, T, C, H, W) - input observations (optional)
            prev_actions: Previous actions (optional)
            prev_state: Previous state (optional)
            mask_: Mask parameter (kept for API compatibility)
            pose: Pose information (B, pose_dim) - for SimpleWorldModel
            last_action: Last action taken (kept for API compatibility)
            text_goal: Text goal (kept for API compatibility)
            goal_image: Goal image (kept for API compatibility)
            return_full_sequence: If True, returns full planned sequence; else just first action
            
        Returns:
            Dictionary containing:
                - 'actions': Random action(s) (B, action_dim) or (B, horizon, action_dim)
                - 'predicted_reward': 0.0 (no prediction for random actions)
                - 'final_state': None or dummy state
        """
        ## compute random actions
        actions = torch.rand((1, self.action_dim), device=pose.device) * 2 - 1  # (1, action_dim) in range [-1, 1]
        
        return {
            'actions': actions,
            'predicted_reward': 0.0,
            'final_state': prev_state if prev_state is not None else None
        }