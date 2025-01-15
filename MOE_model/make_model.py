from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .ExperModel import Static_MoE, Dynamic_MoE, ParallelFFNMoE

def make_model(config: DictConfig):
    if config.model_ckpt:
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='balanced')
        for param in model.parameters():
            param.requires_grad = False

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if config.half:
        model.bfloat16()
    return model, tokenizer


def replace_layer(model, layer_index, original_layer, num_experts, flag):
    ffn_layer = original_layer
    S_moe_layer = Static_MoE(input_dim=4096, hidden_dim=4096, output_dim=4096, num_experts=num_experts)
    D_moe_layer = Dynamic_MoE(input_dim=4096, hidden_dim=4096, output_dim=4096, num_experts=4)
    coe_lambda = 2
    if flag == 0:
        model.model.layers[layer_index].mlp = ParallelFFNMoE(ffn_layer, S_moe_layer, coe_lambda = coe_lambda).cuda()
    elif flag == 1:
        model.model.layers[layer_index].mlp = ParallelFFNMoE(ffn_layer, D_moe_layer, coe_lambda = coe_lambda).cuda()
    elif flag == 2:
        model.model.layers[layer_index].mlp = ParallelFFNMoE(ffn_layer, S_moe_layer, D_moe_layer, coe_lambda = coe_lambda).cuda()


def recover_layer(model, layer_index, original_layer):
    model.model.layers[layer_index].mlp = original_layer
