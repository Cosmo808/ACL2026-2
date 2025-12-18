import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Any
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer
from utils.xnli_snn.neurons import ConditionalLIFNode
from utils.xnli_snn.context_encoder import ContextEncoder


class SNNTokenizer(nn.Module):
    def __init__(
        self,
        char_num: int = 256,
        char_embed_dim: int = 128,
        output_embed_dim: int = 768,   # must match LLM hidden size
        entropy: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.char_embed_dim = char_embed_dim
        self.output_embed_dim = output_embed_dim
        self.device = device
        self.entropy = entropy

        # === 1. Char Embedding ===
        self.char_embedding = nn.Embedding(char_num, char_embed_dim, padding_idx=0)

        # === 2. Shared ANN Encoder (for boundary and reset prediction)===
        # self.context_encoder = ContextEncoder(d_model=char_embed_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(char_embed_dim, char_embed_dim),
            nn.LayerNorm(char_embed_dim),
            nn.GELU(),
        )

        # === 3. Boundary & Reset Predictors (identical structure) ===
        self.boundary_predictor = nn.Sequential(
            nn.Linear(char_embed_dim, 1),
            nn.Tanh(),
        )
        self.reset_predictor = nn.Linear(char_embed_dim, 1)

        # === 4. Conditional SNN Node (multi-step) ===
        self.node = ConditionalLIFNode(step_mode='m')

        # === 5. Projection to LLM space ===
        self.projection = nn.Sequential(
            nn.LayerNorm(char_embed_dim),
            nn.Linear(char_embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_embed_dim),
            nn.LayerNorm(output_embed_dim)
        )

        # === 6. Initialization ===
        self.reset_logits = None
        self.hard_boundaries = None
        self.soft_boundaries = None
        self.token_ids = None
        self.I = None
        self.to(device)

    def forward(
        self,
        inputs: List[str],
        use_hard_boundaries: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            inputs_embeds: (B, L, output_embed_dim) ready for LLM encoder
        """
        # === Step 1: Text → Char Embeddings ===
        char_embs = self.char_embedding(inputs)              # (B, T, char_embed_dim)

        # === Step 2: Contextual encoding (ANN) ===
        hidden = self.context_encoder(char_embs)               # (B, T, ann_hidden_dim)

        # === Step 3: Predict boundary and reset logits ===
        boundary_logits = self.boundary_predictor(hidden).squeeze(-1)  # (B, T)
        self.I = boundary_logits
        reset_logits = self.reset_predictor(hidden).squeeze(-1)        # (B, T)

        # === Step 4: Spiking neuron and reset mask ===
        # SNN input: boundary_logits as (T, B, 1)
        snn_input = boundary_logits.transpose(0, 1).unsqueeze(-1)      # (T, B, 1)

        # Reset mask: from reset_logits, (T, B, 1)
        if self.entropy:
            reset_probs = torch.sigmoid(reset_logits)
            reset_mask = (reset_probs > 0.5).transpose(0, 1).unsqueeze(-1)  # (T, B, 1)
        else:
            reset_mask = None

        # Set reset mask and run SNN in multi-step mode
        self.node.set_reset_mask(reset_mask)
        spikes = self.node(snn_input)                                 # (T, B, 1)

        # === Step 5: Get boundaries===
        soft_boundaries = torch.stack(self.node.past_v).squeeze(-1).transpose(0, 1)  # (B, T)
        hard_boundaries = spikes.squeeze(-1).transpose(0, 1)  # (B, T)

        # === Step 6: Choose boundary type ===
        if use_hard_boundaries:
            boundaries = hard_boundaries.float()
        else:
            boundaries = soft_boundaries

        # === Step 7: Soft grouping → token embeddings ===
        token_embs, token_ids = self._soft_group(hidden, soft_boundaries, hard_boundaries)  # (B, L, ann_hidden_dim), (B, T)

        # === Step 8: Project to LLM space ===
        inputs_embeds = self.projection(token_embs)                   # (B, L, output_embed_dim)

        # Save for external loss (e.g., entropy loss on reset_logits)
        self.soft_boundaries = soft_boundaries.detach()
        self.hard_boundaries = hard_boundaries.detach()
        self.reset_logits = reset_logits.detach()
        self.token_ids = token_ids.detach()

        return inputs_embeds

    def _soft_group(self, char_embs, boundaries: torch.Tensor, hard_boundaries: torch.Tensor):
        """
        Tokenization with:
          - hard_boundaries (spikes) for token segmentation
          - boundaries (membrane potential) for intra-token weighting

        Returns:
            token_embs: (B, K, D)
            token_ids:  (B, T)
        """
        B, T, D = char_embs.shape
        device = char_embs.device

        # Step 1: token segmentation (STRUCTURE) using hard boundaries
        # hard_boundaries: 1 means "start a new token next"
        token_ids = torch.cumsum(hard_boundaries.long(), dim=1)  # (B, T)
        token_ids = torch.roll(token_ids, shifts=1, dims=1)
        token_ids[:, 0] = 0

        K = token_ids.max().item() + 1
        K = min(K, T)
        token_ids = torch.clamp(token_ids, 0, K - 1)

        # Step 2: build token masks (B, T, K)
        token_mask = torch.zeros(B, T, K, device=device)
        token_mask.scatter_(2, token_ids.unsqueeze(-1), 1.0)

        # Step 3: compute intra-token weights using soft boundaries
        weights_raw = torch.relu(boundaries)  # (B, T)
        weights = weights_raw.unsqueeze(-1) * token_mask  # (B, T, K)
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-8
        weights = weights / weights_sum  # (B, T, K)

        # Step 4: weighted aggregation → token embeddings
        token_embs = torch.einsum("btk,btd->bkd", weights, char_embs)  # (B, K, D)

        return token_embs, token_ids
