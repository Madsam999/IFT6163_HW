import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def get_patches_fast(images, cfg):
    from einops import rearrange
    batch_size, height, width, channels = images.shape
    patch_size = cfg.patch_size ## n_patches = 8

    patches = rearrange(images[:,:,:,:3], 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
    if channels > 3:
        ## History stacking in the channel dimension for observations only, not goal images.
        patches = rearrange(images, 'b (h p1) (w p2) (c hs) -> b (h w hs) (p1 p2 c)', p1 = patch_size, p2 = patch_size, hs=cfg.policy.obs_stacking) ## Stack the history in the channel dimension
    return patches


def calc_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B,T,C = x.shape
        # TODONE: 
        ## Provide the block masking logic for the attention head
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        if mask is not None:
          if mask.dim() == 1:
            wei = wei.masked_fill(mask.unsqueeze(0).unsqueeze(1) == 0, float('-inf')) 
          else:
            wei = wei.masked_fill(mask == 0, float('-inf'))
        #wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd=n_embd, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        with torch.profiler.record_function("Self-Attention"):
            out = torch.cat([h(x, mask) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, mask=None):
        x = x + self.sa(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GRP(nn.Module):
    def __init__(self, cfg, mlp_ratio=4):
        super(GRP, self).__init__()
        self._cfg = cfg
        chars = cfg.dataset.chars_list
        cfg.vocab_size = len(chars)

        self.n_embd = cfg.n_embd
        self.patch_size = cfg.patch_size

        patch_dim = self.patch_size * self.patch_size * 3
        self.patch_embedding = nn.Linear(patch_dim, self.n_embd)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.n_embd) * 0.02)

        if not self._cfg.dataset.encode_with_t5:
          self.token_embedding_table = nn.Embedding(cfg.vocab_size, self.n_embd)

        self.blocks = nn.Sequential(*[
          Block(self.n_embd, n_head=cfg.n_head, dropout=cfg.dropout) 
          for _ in range(cfg.n_blocks)
        ])

        self.ln_f = nn.LayerNorm(self.n_embd)

        self.action_space_type = getattr(cfg.policy, 'action_space_type', 'continuous')
        self.total_action_dim = cfg.action_dim * cfg.policy.action_stacking

        if self.action_space_type == 'discrete':
            self.n_bins = 14
            self.head = nn.Linear(self.n_embd, self.total_action_dim * self.n_bins)
        else:
            self.head = nn.Linear(self.n_embd, self.total_action_dim)

        self.apply(self._init_weights)
        # TODONE: 
        ## Provide the logic for the GRP network

        # 4) Transformer encoder blocks

        # 5) Classification MLPk

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, images, goals_txt, goal_imgs, targets=None, pose=None, mask_=False):
        n, c, h, w = images.shape
        device = images.device 
        
        obs_patches = get_patches_fast(images, self._cfg)
        obs_emb = self.patch_embedding(obs_patches) 
        
        patches_g = get_patches_fast(goal_imgs, self._cfg)
        goal_img_emb = self.patch_embedding(patches_g)
        
        if self._cfg.dataset.encode_with_t5:
            goals_e = goals_txt 
        else:
            goals_e = self.token_embedding_table(goals_txt)

      
        if self.training:
            mask_type = torch.randint(0, 3, (1,)).item() 
            if mask_type == 0:
                goals_e = torch.zeros_like(goals_e)
            elif mask_type == 1:
                goal_img_emb = torch.zeros_like(goal_img_emb)

      
        cls_tokens = self.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_tokens, obs_emb, goals_e, goal_img_emb), dim=1)

        seq_len = x.shape[1]
        pos_emb = calc_positional_embeddings(seq_len, self.n_embd)
        pos_emb = pos_emb.to(x.device)
        x += pos_emb.unsqueeze(0)

        mask = None 
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)

        cls_out = x[:, 0, :]
        logits = self.head(cls_out)
        out = logits

        loss = None
        if targets is not None:
            if self.action_space_type == "discrete":
                logits_reshaped = logits.view(n, self.total_action_dim, self.n_bins)
                targets_norm = (targets + 1) / 2
                targets_bin = (targets_norm * (self.n_bins - 1)).round().long()
                targets_bin = torch.clamp(targets_bin, 0, self.n_bins - 1)
                loss = F.cross_entropy(logits_reshaped.permute(0, 2, 1), targets_bin)
            else:
                loss = F.mse_loss(logits, targets)

        return (out, loss) 
    
    def resize_image(self, image):
        """
        Docstring for resize_image
        
        :param self: Description
        :param image: Description
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        import cv2
        import numpy as _np
        img = _np.array(image, dtype=_np.float32)
        img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        return img

    def normalize_state(self, image):
        """
        Docstring for preprocess_state
        
        :param self: Description
        :param image: Description
        self._encode_state = lambda af:   ((af/(255.0)*2.0)-1.0) # encoder: take a float, output an integer
        self._resize_state = lambda sf:   cv2.resize(np.array(sf, dtype=np.float32), (cfg.image_shape[0], cfg.image_shape[1]))  # resize state
        """
        # img = _np.array(image, dtype=_np.float32)
        # img = cv2.resize(img, (self._cfg.image_shape[0], self._cfg.image_shape[1]))
        enc = ((image / 255.0) * 2.0) - 1.0
        # t = _torch.tensor(enc, dtype=_torch.float32, device=self._cfg.device)
        return enc
    
    def preprocess_state(self, image):
        img = self.resize_image(image)
        img = self.normalize_state(img)
        return img

    def preprocess_goal_image(self, image):
        return self.preprocess_state(image)

    def encode_text_goal(self, goal, tokenizer=None, text_model=None):
        import numpy as _np
        import torch as _torch
        if self._cfg.dataset.encode_with_t5:
            if tokenizer is None or text_model is None:
                raise ValueError("tokenizer and text_model must be provided when using T5 encoding")
            # TODONE:    
            ## Provide the logic converting text goal to T5 embedding tensor
            with _torch.no_grad():
                input_ids = tokenizer(goal, return_tensores="pt").input_ids.to(self._cfg.device)

                outputs = text_model.encoder(input_ids=input_ids)
                last_hidden_state = outputs.last_hidden_state
            
            result = _torch.zeros((1, self._cfg.max_block_size, self._cfg.n_embd), d_type=_torch.float32, device=self._cfg.device)
            seq_length = last_hidden_state.shape[1]
            limit = min(seq_length, self._cfg.max_block_size)
            result[:, :limit, :] = last_hidden_state[:, :limit, :]
            return result
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

    def decode_action(self, action_tensor):
        
        """
        Docstring for decode_action
        
        :param self: Description
        :param action_tensor: Description
        self._decode_action = lambda binN: (binN * action_std) + action_mean  # Undo mapping to [-1, 1]
        """
        import torch as _torch
        ## The action tensor is of shape (batch_size, action_dim * action_stacking) so we need to repeat the mean and std per action stacking
        action_mean = _torch.tensor(np.repeat(self._cfg.dataset.action_mean, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        action_std = _torch.tensor(np.repeat(self._cfg.dataset.action_std, self._cfg.policy.action_stacking), dtype=action_tensor.dtype, device=action_tensor.device)
        return (action_tensor * action_std) + action_mean
    
    def encode_action(self, action_float):
        """
        Docstring for encode_action
        
        :param self: Description
        :param action_float: Description
        self._encode_action = lambda af:   (af - action_mean)/(action_std) # encoder: take a float, output an integer
        """
        import torch as _torch
        action_mean = _torch.tensor(self._cfg.dataset.action_mean, dtype=action_float.dtype, device=action_float.device)
        action_std = _torch.tensor(self._cfg.dataset.action_std, dtype=action_float.dtype, device=action_float.device)
        return (action_float - action_mean) / action_std


@torch.no_grad()
def estimate_loss(model, dataset):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(model._cfg.eval_iters)
        for k in range(model._cfg.eval_iters):
            X, x_pose, x_goal, x_goal_img, Y = dataset.get_batch_grp(split, model._cfg, model._cfg.batch_size)
            logits, loss = model(X, x_goal, x_goal_img, Y, pose=x_pose)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
