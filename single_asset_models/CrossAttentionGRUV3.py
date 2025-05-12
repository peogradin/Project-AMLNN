import torch
import torch.nn as nn

class CrossAttentionGRUV3(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hidden_layers, out_dim, dropout, num_heads):
        super().__init__()
        # 1) GRU with inter‐layer dropout
        self.gru_model = nn.GRU(
            in_dim, hidden_dim,
            num_layers=num_hidden_layers,
            dropout=dropout if num_hidden_layers>1 else 0.0,
            batch_first=True
        )

        # 2) Input projection + BatchNorm
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        # 3) Cross‐attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=hidden_dim,
            vdim=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        # 4) Fuse & normalize
        self.fuse_fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bn_fuse = nn.BatchNorm1d(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 5) Final output
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self._stored_attn_weights = None

    def get_attn_weights(self):
        return self._stored_attn_weights

    def forward(self, x):
        # x: (batch, time, in_dim)
        B, T, _ = x.size()

        # --- 1) GRU path ---
        gru_out, _ = self.gru_model(x)     # (B, T, H)
        gru_out = self.dropout(gru_out)

        # --- 2) Project + BatchNorm on each time‐slice ---
        x_proj = self.input_proj(x)        # (B, T, H)
        #flat_proj = x_proj.contiguous().view(B*T, -1)
        #bn_proj   = self.bn_input(flat_proj).view(B, T, -1)

        # --- 3) Cross attention ---
        attn_out, attn_w = self.cross_attn(
            query=gru_out,
            key=x_proj,
            value=x_proj
        )
        self._stored_attn_weights = attn_w.detach()

        # --- 4) Fuse & normalize ---
        combined = attn_out + gru_out  # (B, T, 2H)
        #fused    = self.fuse_fc(combined)                  # (B, T, H)

        #flat_fuse = fused.contiguous().view(B*T, -1)
        #bn_fuse   = self.bn_fuse(flat_fuse).view(B, T, -1)
        norm_combined = self.layer_norm(combined)

        # --- 5) Select final step, dropout & output ---
        final_step = norm_combined[:, -1, :]   # (B, H)
        final_step = self.dropout(final_step)
        output     = self.fc(final_step)   # (B, out_dim)

        return output
