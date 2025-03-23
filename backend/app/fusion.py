# fusion.py
import torch
from torch import nn

class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Updated input dimensions
        self.audio_encoder = nn.Linear(512, 128)  # Matches LSTM output size
        self.visual_encoder = nn.Linear(8, 128)    # 7 emotions + 1 motion
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # negative, neutral, positive
        )

    def forward(self, audio_input, visual_input):
        # Ensure proper dimensions [batch_size, features]
        if audio_input.dim() == 3:
            audio_input = audio_input.squeeze(1)
        if visual_input.dim() == 3:
            visual_input = visual_input.squeeze(1)

        # Project features
        audio_proj = self.audio_encoder(audio_input)  # [1, 128]
        visual_proj = self.visual_encoder(visual_input)  # [1, 128]

        # Attention expects [seq_len, batch_size, embed_dim]
        attn_output, _ = self.attention(
            audio_proj.unsqueeze(0),  # [1, 1, 128]
            visual_proj.unsqueeze(0),  # [1, 1, 128]
            visual_proj.unsqueeze(0)
        )

        # Combine features
        combined = torch.cat([audio_proj, visual_proj], dim=1)
        return self.classifier(combined)