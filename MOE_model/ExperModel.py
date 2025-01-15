import torch
import torch.nn as nn

# from sentence_transformers import SentenceTransformer
# rounter_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# 定义专家模块，每个专家可以是一个简单的全连接层
class LowRankExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, rank):
        """
        Expert module with one hidden layer, ReLU activation, and low-rank decomposition.

        Args:
            input_dim (int): Dimensionality of the input features.
            hidden_dim (int): Dimensionality of the hidden layer.
            output_dim (int): Dimensionality of the output features.
            rank (int): Rank for low-rank decomposition.
        """
        super(LowRankExpert, self).__init__()

        # Low-rank decomposition for hidden layer
        self.hidden_layer_u = nn.Linear(input_dim, rank, bias=False)  # First projection
        self.hidden_layer_v = nn.Linear(rank, hidden_dim, bias=True)  # Second projection

        self.activation = nn.ReLU()

        # Low-rank decomposition for output layer
        self.output_layer_u = nn.Linear(hidden_dim, rank, bias=False)  # First projection
        self.output_layer_v = nn.Linear(rank, output_dim, bias=True)  # Second projection

    def forward(self, x):
        """
        Forward pass through the low-rank expert module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Low-rank hidden layer
        x = self.hidden_layer_u(x)  # Project to low-rank space
        x = self.hidden_layer_v(x)  # Project to hidden_dim space
        x = self.activation(x)

        # Low-rank output layer
        x = self.output_layer_u(x)  # Project to low-rank space
        x = self.output_layer_v(x)  # Project to output_dim space
        return x


# 定义 MoE 层，包含门控机制和多个专家
class Static_MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(Static_MoE, self).__init__()
        self.experts = nn.ModuleList(LowRankExpert(input_dim, hidden_dim, output_dim, rank=512) for _ in range(num_experts))
        # self.expert = LowRankExpert(input_dim, hidden_dim, output_dim, rank=512)

    def forward(self, x):
        output = torch.zeros(x.shape).cuda()  # 创建一个与输入形状相同的零张量来存储最终输出
        # 对于每个样本，使用不同的专家处理
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # 每个专家处理一个输入（注意需要增加batch维度）
            output += expert_output# 将输出保存到对应位置
        return output / len(self.experts)



class Dynamic_MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=4):
        super(Dynamic_MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(LowRankExpert(input_dim, hidden_dim, output_dim, rank=512) for _ in range(num_experts))
        self.gate = nn.Linear(input_dim, num_experts)  # gate decision, W_g


    def forward(self, x):
        gate_output = torch.softmax(self.gate(torch.cat([x],  dim=-1)), dim=-1)  # 生成每个专家的权重
        ## soft-control
        output = 0
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)
            output += gate_output[:, i:i+1] * expert_output  # weighted the results of each expert
        return output



# 定义并行处理的函数
class ParallelFFNMoE(nn.Module):
    def __init__(self, ffn, moe_layer_1, moe_layer_2=None, coe_lambda=1):
        super(ParallelFFNMoE, self).__init__()
        self.ffn = ffn
        self.moe_layer_1 = moe_layer_1
        self.moe_layer_2 = moe_layer_2
        self.coe_lambda = coe_lambda

    def forward(self, x, id):
        # 创建一个空列表，用于存放每个token计算的结果
        final_output = torch.zeros_like(x)
        for i in range(x.size(1)):
            token = x[:, i, :]
            if i < id and x.size(1) != 1:
                token_ffn_output = self.ffn(token)  # FFN处理
                final_output[:, i:i+1, :] = token_ffn_output.unsqueeze(dim=1)
            else:
                token_ffn_output = self.ffn(token)  # FFN处理
                if self.moe_layer_2:
                    token_moe_output_1 = self.moe_layer_1(token)  # MoE处理
                    token_moe_output_2 = self.moe_layer_2(token)  # MoE处理
                    final_output[:, i:i+1, :] = (token_ffn_output.unsqueeze(dim=1) +
                                                 (self.coe_lambda * token_moe_output_1.unsqueeze(dim=1)) +
                                                 (self.coe_lambda * token_moe_output_2.unsqueeze(dim=1)))
                else:
                    token_moe_output_1 = self.moe_layer_1(token)  # MoE处理
                    final_output[:, i:i + 1, :] = (token_ffn_output.unsqueeze(dim=1) +
                                                   (self.coe_lambda * token_moe_output_1.unsqueeze(dim=1)))

        return final_output

