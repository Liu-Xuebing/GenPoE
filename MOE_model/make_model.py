from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from .ExperModel import MoE, ParallelFFNMoE

def make_model(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='auto')

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if config.half:
        model.bfloat16()
    return model, tokenizer


def replace_layer(model, layer_index, original_layer, num_experts):
    moes = MoE(num_experts=num_experts)
    if "Llama-2-7b-hf" in model.config._name_or_path:
        model.model.layers[layer_index].mlp = ParallelFFNMoE(original_layer, moes).to(next(original_layer.parameters()).device)
    elif "gpt-j-6B" in model.config._name_or_path:
        model.transformer.h[layer_index].mlp = ParallelFFNMoE(original_layer, moes).to(next(original_layer.parameters()).device)


def recover_layer(model, layer_index, original_layer):
    model.model.layers[layer_index].mlp = original_layer
