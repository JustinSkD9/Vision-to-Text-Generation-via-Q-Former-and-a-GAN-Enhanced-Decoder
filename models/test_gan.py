import torch.nn as nn

class TestGAN(nn.Module):
    def __init__(self, vocab_size, hidden_size=512, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        attn_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        cls_repr = x[:, 0, :]
        logits = self.fc(cls_repr)
        return logits
