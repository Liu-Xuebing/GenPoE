import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, embed_dim, rank, hidden_dim, output_dim, input_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank

        total_params = (
            rank * input_dim +     # delta for hidden_layer_u
            hidden_dim * rank +    # delta for hidden_layer_v
            rank * hidden_dim +    # delta for output_layer_u
            output_dim * rank      # delta for output_layer_v
        )
        self.generator = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, total_params)
        )

    def forward(self, x_embed):  # x_embed shape: [batch_size, embed_dim]
        x_embed = x_embed.mean(dim=1)
        delta_vector = self.generator(x_embed)  # shape: [batch_size, total_params]
        return delta_vector