from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .ExperModel import MoE, ParallelFFNMoE
from .RouterModel import Router

def make_model(config: DictConfig):
    if config.model_ckpt:
        pass
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # tokenizer.add_tokens(["[SCORE]"])
        model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='auto')
        # model.resize_token_embeddings(len(tokenizer))  # 更新词嵌入
        for param in model.parameters():
            param.requires_grad = False

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if config.half:
        model.bfloat16()
    return model, tokenizer


def replace_layer(model, layer_index, original_layer, num_experts):
    ffn_layer = original_layer
    moes = MoE(num_experts=num_experts)
    model.model.layers[layer_index].mlp = ParallelFFNMoE(ffn_layer, moes).to(next(ffn_layer.parameters()).device)


def recover_layer(model, layer_index, original_layer):
    model.model.layers[layer_index].mlp = original_layer
