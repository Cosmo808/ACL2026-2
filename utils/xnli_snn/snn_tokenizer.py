import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from spikingjelly.activation_based import base, neuron, functional, surrogate, layer
from utils.xnli_snn.neurons import ConditionalLIFNode


class SNNTokenizer(nn.Module):
    def __init__(
        self,
        char_embed_dim: int = 128,
        ann_hidden_dim: int = 256,
        output_embed_dim: int = 768,   # must match LLM hidden size
        max_char_len: int = 256,       # max UTF-8 byte length
        entropy: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.char_embed_dim = char_embed_dim
        self.ann_hidden_dim = ann_hidden_dim
        self.output_embed_dim = output_embed_dim
        self.max_char_len = max_char_len
        self.device = device
        self.entropy = entropy

        # === 1. Char Embedding (256 = full byte range) ===
        self.char_embedding = nn.Embedding(256, char_embed_dim, padding_idx=0)

        # === 2. Shared ANN Encoder (for boundary and reset prediction)===
        self.context_encoder = nn.Sequential(
            layer.Linear(char_embed_dim, ann_hidden_dim, step_mode='m'),
            layer.BatchNorm1d(ann_hidden_dim),
            neuron.IFNode(step_mode='m'),
            layer.Linear(ann_hidden_dim, ann_hidden_dim, step_mode='m'),
            layer.BatchNorm1d(ann_hidden_dim),
            neuron.IFNode(step_mode='m'),
        )

        # === 3. Boundary & Reset Predictors (identical structure) ===
        self.boundary_predictor = layer.Linear(ann_hidden_dim, 1, step_mode='m')
        self.reset_predictor = layer.Linear(ann_hidden_dim, 1, step_mode='m')

        # === 4. Conditional SNN Node (multi-step, reset-controlled) ===
        self.node = ConditionalLIFNode(step_mode='m', v_reset=0., tau=2.)

        # === 5. Projection to LLM space ===
        self.projection = nn.Sequential(
            nn.Linear(ann_hidden_dim, output_embed_dim),
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
        texts: List[str],
        use_hard_boundaries: bool = False,
    ) -> torch.Tensor:
        """
        Returns:
            inputs_embeds: (B, L, output_embed_dim) ready for LLM encoder
        """
        # === Step 1: Text → UTF-8 byte IDs → Char Embeddings ===
        byte_ids, boundary_mask = self._text_to_byte_ids(texts)               # (B, T)
        char_embs = self.char_embedding(byte_ids)              # (B, T, char_embed_dim)

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
        soft_boundaries = torch.sigmoid(boundary_logits)  # (B, T)
        hard_boundaries = spikes.squeeze(-1).transpose(0, 1)  # (B, T)

        # === Step 6: Apply UTF-8 boundary mask ===
        soft_boundaries = soft_boundaries * boundary_mask + (-1e4) * (1 - boundary_mask)
        hard_boundaries = hard_boundaries * boundary_mask

        # === Step 7: Choose boundary type ===
        if use_hard_boundaries:
            boundaries = hard_boundaries.float()
        else:
            boundaries = soft_boundaries

        # Ensure first token always starts
        boundaries[:, 0] = 1.0

        # === Step 8: Soft grouping → token embeddings ===
        token_embs, token_ids = self._soft_group(hidden, boundaries)  # (B, L, ann_hidden_dim), (B, T)

        # === Step 9: Project to LLM space ===
        inputs_embeds = self.projection(token_embs)                   # (B, L, output_embed_dim)

        # Save for external loss (e.g., entropy loss on reset_logits)
        self.soft_boundaries = soft_boundaries
        self.hard_boundaries = hard_boundaries
        self.reset_logits = reset_logits
        self.token_ids = token_ids

        return inputs_embeds

    def _soft_group(self, char_embs: torch.Tensor, boundaries: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Differentiable soft grouping via cumulative boundary-based segmentation.
        char_embs: (B, T, D)
        boundaries: (B, T) ∈ [0,1], higher = more likely start of new token
        Returns: (B, K, D) token-level embeddings (K <= T)
        """
        B, T, D = char_embs.shape

        # Cumulative sum to assign token IDs
        cum_bound = torch.cumsum(boundaries, dim=1)  # (B, T)
        token_ids = (cum_bound - 1).long()           # (B, T), 0-indexed

        # Max token count in batch
        K = token_ids.max().item() + 1
        K = min(K, T)

        # One-hot assignment (B, T, K)
        token_ids = torch.clamp(token_ids, 0, K - 1)
        assignment = torch.zeros(B, T, K, device=char_embs.device)
        assignment.scatter_(2, token_ids.unsqueeze(-1), 1.0)

        # Normalize weights per token
        weights = assignment / (assignment.sum(dim=1, keepdim=True) + 1e-8)  # (B, T, K)

        # Weighted sum: (B, K, D)
        token_embs = torch.einsum('bt k,bt d->bk d', weights, char_embs)
        return token_embs, token_ids

    def _text_to_byte_ids(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert list of texts to:
          - byte_ids: (B, T) UTF-8 byte IDs (0-255)
          - char_start_mask: (B, T) bool, True = this byte is the start of a UTF-8 character
        """
        batch_bytes = []
        batch_masks = []

        for text in texts:
            byte_list = []
            mask_list = []

            for char in text:
                # Encode single char to UTF-8 bytes
                char_bytes = char.encode('utf-8')  # e.g., 'é' → b'\xc3\xa9'
                char_byte_vals = [b for b in char_bytes]

                # First byte is start of character → True
                # Following bytes are continuation → False
                char_mask = [1] + [0] * (len(char_byte_vals) - 1)

                byte_list.extend(char_byte_vals)
                mask_list.extend(char_mask)

            # Truncate or pad
            if len(byte_list) > self.max_char_len:
                byte_list = byte_list[:self.max_char_len]
                mask_list = mask_list[:self.max_char_len]
            else:
                pad_len = self.max_char_len - len(byte_list)
                byte_list.extend([0] * pad_len)
                mask_list.extend([0] * pad_len)  # padding is not char start

            batch_bytes.append(torch.tensor(byte_list, dtype=torch.long, device=self.device))
            batch_masks.append(torch.tensor(mask_list, dtype=torch.long, device=self.device))

        return (
            torch.stack(batch_bytes),  # (B, T)
            torch.stack(batch_masks)  # (B, T), True = valid boundary position
        )