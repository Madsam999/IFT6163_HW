import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Independent, kl_divergence
import numpy as np

def symlog(x):
    """
    Symmetric log transformation.
    Squashes large values while preserving sign and small values.
    y = sign(x) * ln(|x| + 1)
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

class GRPBase(nn.Module):
    """Base class for GRP models"""
    def __init__(self, cfg):
        super(GRPBase, self).__init__()
        self._cfg = cfg
        print(self._cfg.dataset.encode_with_t5, " dreamerV3.py")

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODO:    
            ## Provide the logic converting text goal to T5 embedding tensor
            tokens = tokenizer(
                goal, 
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self._cfg.max_block_size
            ).to(self._cfg.device)

            # 2. Forward pass through the T5 encoder only
            with _torch.no_available_gradients() if not _torch.is_grad_enabled() else _torch.enable_grad():
                outputs = text_model.get_encoder()(**tokens)
                # last_hidden_state shape: (1, max_block_size, n_embd)
                embeddings = outputs.last_hidden_state
            
            return embeddings
        else:
            pad = " " * self._cfg.max_block_size
            goal_ = goal[:self._cfg.max_block_size] + pad[len(goal):self._cfg.max_block_size]
            try:
                stoi = {c: i for i, c in enumerate(self._cfg.dataset.chars_list)}
                ids = [stoi.get(c, 0) for c in goal_]
            except Exception:
                ids = [0] * self._cfg.max_block_size
            return _torch.tensor(_np.expand_dims(_np.array(ids, dtype=_np.int64), axis=0), dtype=_torch.long, device=self._cfg.device)

    def process_text_embedding_for_buffer(self, goal, tokenizer=None, text_model=None):
        """
        Process text goal embedding for storing in the circular buffer.
        Returns a numpy array of shape (max_block_size, n_embd) without batch dimension.
        """
        import numpy as _np
        if tokenizer is None or text_model is None:
            raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
        
        goal_ = _np.zeros((self._cfg.max_block_size, self._cfg.n_embd), dtype=_np.float32)
        input_ids = tokenizer(goal, return_tensors="pt").input_ids
        goal_t = text_model.encoder(input_ids).last_hidden_state.detach().cpu().numpy()
        goal_[:len(goal_t[0]), :] = goal_t[0][:self._cfg.max_block_size]
        return goal_


    def resize_image(self, image):
        """Resize image to match model input size"""
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """Normalize image to [-1, 1] range"""
        enc = ((image / 255.0) * 2.0) - 1.0
        return enc
    
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        """Preprocess goal image"""
        return self.preprocess_state(image)

    def decode_action(self, action_tensor):
        """Decode normalized actions to original action space"""
        import torch as _torch
        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * (action_std)) + action_mean

    def encode_action(self, action_float):
        """Encode actions to normalized space [-1, 1]"""
        import torch as _torch
        ## If the action_float has length greater than action_dim then use stacking otherwise just use normal standardiaztion vectors
        if action_float.shape[1] == len(self._cfg.dataset.action_mean):
            action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
            action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
            return (action_float - action_mean) / (action_std)  

        action_mean = _torch.tensor(np.repeat([self._cfg.dataset.action_mean], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                   dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(np.repeat([self._cfg.dataset.action_std], self._cfg.policy.action_stacking, axis=0).flatten(), 
                                  dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / (action_std)
    
    def decode_pose(self, pose_tensor):
        """
        Docstring for decode_pose
        
        :param self: Description
        :param pose_tensor: Description
        self._decode_state = lambda sinN: (sinN * state_std) + state_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_tensor.dtype, device=pose_tensor.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_tensor.dtype, device=pose_tensor.device)
        return (pose_tensor * (pose_std)) + pose_mean
    
    def encode_pose(self, pose_float):
        """
        Docstring for encode_pose
        
        :param self: Description
        :param pose_float: Description
        self._encode_pose = lambda pf:   (pf - pose_mean)/(pose_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        pose_mean = _torch.tensor(self._cfg.dataset.pose_mean, dtype=pose_float.dtype, device=pose_float.device)
        pose_std = _torch.tensor(self._cfg.dataset.pose_std, dtype=pose_float.dtype, device=pose_float.device)
        return (pose_float - pose_mean) / (pose_std)

class DreamerV3(GRPBase):
    def __init__(self, 
                 obs_shape=(3, 128, 128),  # Updated default to match your error
                 action_dim=6, 
                 stoch_dim=32, 
                 discrete_dim=32, 
                 deter_dim=512, 
                 hidden_dim=512, cfg=None):
        # TODO: Part 3.1 - Initialize DreamerV3 architecture
        ## Define encoder, RSSM components (GRU, prior/posterior nets), and decoder heads
        super().__init__(cfg)

        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        self.deter_dim = deter_dim
        self.hidden_dim = hidden_dim
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            dummy_img = torch.zeros(1, *obs_shape)
            self.embed_dim = self.encoder(dummy_img).shape[1]

        self.gru = nn.GRUCell(self.stoch_dim * self.discrete_dim + self.action_dim, self.deter_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(self.deter_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim * self.discrete_dim)
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(self.deter_dim + self.embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.stoch_dim * self.discrete_dim)
        )

        self.decoder_input = nn.Linear(self.deter_dim + self.stoch_dim * self.discrete_dim, self.embed_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, int(np.sqrt(self.embed_dim//256)), int(np.sqrt(self.embed_dim//256)))),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.SiLU(),
            nn.ConvTranspose2d(32, obs_shape[0], kernel_size=4, stride=2)
        )

        # 4. Prediction Heads
        self.reward_head = nn.Sequential(
            nn.Linear(self.deter_dim + self.stoch_dim * self.discrete_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        self.continue_head = nn.Sequential(
            nn.Linear(self.deter_dim + self.stoch_dim * self.discrete_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    # ... [Helper methods same as before] ...

    def get_initial_state(self, batch_size, device):
        return {
            'h': torch.zeros(batch_size, self.deter_dim, device=device),
            'z': torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device),
            'z_probs': torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        }

    def sample_stochastic(self, logits, training=True):
        # TODO: Part 3.1 - Implement stochastic sampling
        ## Sample from discrete categorical distribution using logits
        shape = logits.shape
        logits = logits.view(-1, self.stoch_dim, self.discrete_dim)

        if training:
            z = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=1)
        else:
            idx = torch.argmax(logits, dim=-1)
            z = F.one_hot(idx, num_classes=self.discrete_dim).float()
        z_probs = F.softmax(logits, dim=-1)

        z = z.view(shape)
        return z, z_probs

    def rssm_step(self, prev_state, action, embed=None):
        # TODO: Part 3.1 - Implement RSSM step
        ## Update deterministic state (h) with GRU, compute prior and posterior distributions
        gru_input = torch.cat([prev_state['z'], action], dim=-1)
        h = self.gru(gru_input, prev_state['h'])
        
        prior_logits = self.prior_net(h)
        prior_z, prior_probs = self.sample_stochastic(prior_logits)
        prior_state = {'h': h, 'z': prior_z, 'z_probs': prior_probs, 'logits': prior_logits}
        
        if embed is not None:
            post_input = torch.cat([h, embed], dim=-1)
            post_logits = self.posterior_net(post_input)
            post_z, post_probs = self.sample_stochastic(post_logits)
            post_state = {'h': h, 'z': post_z, 'z_probs': post_probs, 'logits': post_logits}
            return post_state, prior_state
            
        return prior_state, prior_state

    def forward(self, observations, prev_actions=None, prev_state=None,
                mask_=True, pose=None, last_action=None,
                text_goal=None, goal_image=None):
        # TODO: Part 3.2 - Implement DreamerV3 forward pass
        ## Encode images, unroll RSSM, and compute reconstructions and heads
        B, T, C, H, W = observations.shape
        device = observations.device
        
        obs_flat = observations.view(B * T, C, H, W)
        embeds_flat = self.encoder(obs_flat)
        embeds = embeds_flat.view(B, T, -1)
        
        if prev_state is None:
            prev_state = self.get_initial_state(B, device)
            
        if prev_actions is None:
            prev_actions = torch.zeros(B, T, self.action_dim, device=device)

        posts, priors = [], []
        curr_state = prev_state
        
        for t in range(T):
            post, prior = self.rssm_step(curr_state, prev_actions[:, t], embeds[:, t])
            posts.append(post)
            priors.append(prior)
            curr_state = post
            
        stacked_h = torch.stack([p['h'] for p in posts], dim=1) # (B, T, deter_dim)
        stacked_z = torch.stack([p['z'] for p in posts], dim=1) # (B, T, stoch_dim * discrete_dim)
        features = torch.cat([stacked_h, stacked_z], dim=-1)    # (B, T, deter+stoch)
        
        features_flat = features.view(B * T, -1)
        dec_in = self.decoder_input(features_flat)
        recons_flat = self.decoder(dec_in)
        # Resize to match target shape if necessary
        if recons_flat.shape[-2:] != (H, W):
             recons_flat = F.interpolate(recons_flat, size=(H, W), mode='bilinear')
        recons = recons_flat.view(B, T, C, H, W)
        
        rewards = self.reward_head(features_flat).view(B, T, 1)
        continues = self.continue_head(features_flat).view(B, T, 1)
        
        posts_logits = torch.stack([p['logits'] for p in posts], dim=1)
        priors_logits = torch.stack([p['logits'] for p in priors], dim=1)

        return {
            'reconstructions': recons,
            'rewards': rewards,
            'continues': continues,
            'posts_logits': posts_logits,
            'priors_logits': priors_logits
        }

    # [Imagine method remains mostly the same, ensuring valid input shapes for heads]
    def preprocess_state(self, image):
        """Preprocess observation image"""
        img = self.resize_image(image)
        img = self.normalize_state(img)
        ## Change numpy array from channel-last to channel-first
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        # img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        return img
    
    def compute_loss(self, output, images, rewards, dones, device):
        # 1. Align time steps
        T_out = output['rewards'].shape[1]
        images = images[:, :T_out]
        rewards = rewards[:, :T_out]
        dones = dones[:, :T_out]

        # 2. Symlog Regression (Crucial for DreamerV3 stability)
        # We apply symlog to the targets to squash the scale of MSE
        recon_loss = F.mse_loss(output['reconstructions'], symlog(images))
        reward_loss = F.mse_loss(output['rewards'].squeeze(-1), symlog(rewards))
    
        continues_target = (1.0 - dones.float())
        continue_loss = F.binary_cross_entropy_with_logits(output['continues'].squeeze(-1), continues_target)

        # 3. Proper Distribution Handling
        B, T, _ = output['posts_logits'].shape
        post_logits = output['posts_logits'].view(B, T, self.stoch_dim, self.discrete_dim)
        prior_logits = output['priors_logits'].view(B, T, self.stoch_dim, self.discrete_dim)

        # Create the standard distributions for the main KL calculation
        post_dist = Independent(OneHotCategorical(logits=post_logits), 1)
        prior_dist = Independent(OneHotCategorical(logits=prior_logits), 1)

        # 4. KL Balancing (The Detach Fix)
        # To detach a distribution, you must create a NEW distribution 
        # using detached logits.

        # For dyn_loss: Prior follows a fixed Posterior
        post_dist_detached = Independent(OneHotCategorical(logits=post_logits.detach()), 1)
        dyn_loss = kl_divergence(post_dist_detached, prior_dist).mean()

        # For rep_loss: Posterior stays near a fixed Prior
        prior_dist_detached = Independent(OneHotCategorical(logits=prior_logits.detach()), 1)
        rep_loss = kl_divergence(post_dist, prior_dist_detached).mean()

        # 5. Free Nats / Clipping
        # Instead of a hard floor at 1.0 (which kills gradients), we use a small free_nats 
        # value or simply let the KL optimize naturally. DreamerV3 typically uses 1.0 
        # as a "free nats" threshold for the raw KL before averaging.
        free_nats = torch.tensor(1.0, device=device)
        dyn_loss = torch.max(dyn_loss, free_nats)
        rep_loss = torch.max(rep_loss, free_nats)

        # 6. Final Weighted Loss
        # Standard V3 coefficients: pred=1.0, dyn=0.5, rep=0.1
        pred_coeff = self.cfg.loss_coeffs.pred_coeff if self.cfg else 1.0
        dyn_coeff = self.cfg.loss_coeffs.dyn_coeff if self.cfg else 0.5
        rep_coeff = self.cfg.loss_coeffs.rep_coeff if self.cfg else 0.1

        total_loss = (pred_coeff * (recon_loss + reward_loss + continue_loss)) + \
                 (dyn_coeff * dyn_loss) + \
                 (rep_coeff * rep_loss)
                 
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'dyn_loss': dyn_loss.item(),
            'rep_loss': rep_loss.item()
        }