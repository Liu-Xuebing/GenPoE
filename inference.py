import os,json
from data.base import make_Validation_loader
import numpy as np
from tqdm import tqdm, trange
from utils import cross_entropy, cal_EM_F1, tansfer_to_scftmax
from MOE_model.make_model import make_model, replace_layer
from MOE_model.hypernetwork import HyperNetwork
import hydra
import torch


metrics = {
        "base_EM": [],
        "base_F1": [],
        "dir_EM": [],
        "dir_F1": [],
        "pro_EM": [],
        "pro_F1": []}


def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn

def EM_F1_of_model_generation(input_id, input_id_len, model, tok, answers):
    output = model.generate(input_id, max_new_tokens=10, temperature=0.0, top_p=1.0, top_k=-1,
                                 do_sample=False)
    predict = tok.decode(output[0], skip_special_tokens=True)
    EM, F1 = cal_EM_F1(predict[len(input_id_len):].strip().split('\n')[0], answers)
    return EM, F1



def valid_o(original_model, tok, valid_loader):
    original_model.eval()
    for tuples in tqdm(valid_loader, desc="Valid"):
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers, activate_sentence = tuples
        base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, original_model, tok, answers)
        dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, original_model, tok, answers)
        pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, original_model, tok, answers)

        for key, value in zip(metrics.keys(), [base_EM, base_F1, dir_EM, dir_F1, pro_EM, pro_F1]):
            metrics[key].append(value)

        for key, value in metrics.items():
            print(key, len(metrics[key]), np.mean(metrics[key]) * 100)

    return metrics



def valid(config, hypernetwork, model, tok, valid_loader, weights):
    hypernetwork.eval()
    model.eval()

    for tuples in tqdm(valid_loader, desc="Valid"):
        weights = tansfer_to_scftmax(weights)
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers, activate_sentence = tuples
        input_embeds = model.model.embed_tokens(
            activate_sentence['input_ids'])  # shape(batchsize, length, embedding_dim:4096)
        delta = hypernetwork(input_embeds)
        with torch.no_grad():
            hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, weights, delta))
            base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, model, tok, answers)
            hook.remove()

            hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(direct_input.size(1)-base_input.size(1), weights, delta))
            dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, model, tok, answers)
            hook.remove()
            #
            hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(prompt_input.size(1)-base_input.size(1), weights, delta))
            pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, model, tok, answers)
            hook.remove()

        for key, value in zip(metrics.keys(), [base_EM, base_F1, dir_EM, dir_F1, pro_EM, pro_F1]):
            metrics[key].append(value)

        for key, value in metrics.items():
            print(key, len(metrics[key]), np.mean(metrics[key]) * 100)

    return metrics



@hydra.main(config_path="config", config_name="config")
def main(config):
    hypernetwork = HyperNetwork(4096, 512, 4096, 4096, 4096).cuda()
    hypernetwork.load_state_dict(torch.load(config.hypernetwork_ckpt.format(config.model_name.split("/")[-1], config.data_name)))  # 加载参数

    model, tok = make_model(config)
    # original_layer = model.model.layers[config.single_layer].mlp
    # replace_layer(model, config.single_layer, original_layer, 1)

    for name, param in model.named_parameters():
        param.requires_grad = False

    valid_loader = make_Validation_loader(config, tok)
    # metrics = valid(config , hypernetwork, model, tok, valid_loader,[1.0])
    metrics = valid_o(model, tok, valid_loader)
    for key, value in metrics.items():
        print(key, len(metrics[key]), np.mean(metrics[key]) * 100)



if __name__ == '__main__':
    main()