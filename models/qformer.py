import torch
from transformers import Blip2QFormerConfig, Blip2QFormerModel

class QFormerEncoder(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_size=1024, num_query_tokens=32,
                 num_hidden_layers=6, num_attention_heads=16):
        super().__init__()
        self.query_tokens = torch.nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        self.image_proj = (torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_size)
        ) if input_dim != hidden_size else torch.nn.Identity())

        config = Blip2QFormerConfig(
            encoder_hidden_size=hidden_size,
            hidden_size=hidden_size,
            num_query_tokens=num_query_tokens,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1
        )
        self.qformer = Blip2QFormerModel(config)

    def forward(self, image_embeds):
        image_embeds = self.image_proj(image_embeds)
        query_tokens = self.query_tokens.expand(image_embeds.size(0), -1, -1).to(image_embeds.device)
        outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=None,
            return_dict=True
        )
        return outputs.last_hidden_state  # [B, num_query_tokens, hidden_size]
