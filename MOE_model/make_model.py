from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .ExperModel import MoE, ParallelFFNMoE

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


def replace_layer(model, layer_index, original_layer, num_experts):
    ffn_layer = original_layer
    moe_layer = MoE(input_dim=4096, hidden_dim=4096, output_dim=4096, num_experts=num_experts)
    model.model.layers[layer_index].mlp = ParallelFFNMoE(ffn_layer, moe_layer, 1).cuda()


def recover_layer(model, layer_index, original_layer):
    model.model.layers[layer_index].mlp = original_layer
