import os
from signal import pthread_sigmask
from sympy.physics.quantum.identitysearch import lr_op
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm, trange
from data.base import make_Training_loader, make_Validation_loader
from utils import cal_anchor_embedding, cross_entropy, cal_EM_F1, find_subsequence_index, sub_dataset_expert
from MOE_model.make_model import make_model, replace_layer, recover_layer
import hydra
from torch.optim import AdamW
import torch
from time import sleep
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import copy
import json
from sentence_transformers import SentenceTransformer
rounter_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def create_pre_hook_fn(id, weights):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights)
    return pre_hook_fn



def EM_F1_of_model_generation(input_id, input_id_len, model, tok, answers):
    output = model.generate(input_id, max_new_tokens=10, temperature=0.0, top_p=1.0, top_k=-1,
                                 do_sample=False)
    predict = tok.decode(output[0], skip_special_tokens=True)
    EM, F1 = cal_EM_F1(predict[len(input_id_len):].strip().split('\n')[0], answers)
    return EM, F1



def train(config, model, tok, train_loader, optimizer, scheduler, layer):
    model.train()
    for epoch in range(config.epochs):
        running_losses = 0.0
        for tuples in tqdm(train_loader, desc="Train"):
            input_id, sentences = tuples
            optimizer.zero_grad()
            hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, [None]))
            logits = model(**input_id)["logits"]
            loss = cross_entropy(logits, input_id["labels"])
            loss.backward()
            optimizer.step()
            running_losses += loss.item()
            hook.remove()
        scheduler.step()
        print(f"Training Loss: {running_losses:.4f}")



def valid(model, tok, valid_loader, layer):
    model.eval()
    for tuples in tqdm(valid_loader, desc="Valid"):
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers = tuples
        with torch.no_grad():
            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, 1))
            base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, model, tok, answers)
            # hook.remove()

            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(direct_input.size(1)-base_input.size(1), 1))
            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, [1]))
            dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, model, tok, answers)
            # hook.remove()

            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(prompt_input.size(1)-base_input.size(1), 1))
            # hook = model.model.layers[layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, [1]))
            pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, model, tok, answers)
            # hook.remove()

    return base_EM, base_F1, dir_EM, dir_F1, pro_EM, pro_F1




@hydra.main(config_path="config", config_name="config")
def main(config):
    model, tok = make_model(config)
    metrics = {
        "base_EM": [],
        "base_F1": [],
        "dir_EM": [],
        "dir_F1": [],
        "pro_EM": [],
        "pro_F1": []}


    # for i in trange(1500):
    #     for j in range(8):
    #         if os.path.exists(os.path.join(config.save_path, '{}_{}.pth'.format(i, j))):
    #             continue
    #         with open(os.path.join(config.train_path, '{}_{}.json'.format(i, j))) as f:
    #             data = json.load(f)
    #         if len(data) == 0:
    #             continue
    #
    #         original_layer = model.model.layers[config.single_layer].mlp
    #         replace_layer(model, config.single_layer, original_layer, config.num_experts)
    #
    #         optimizer = AdamW(model.parameters(), lr=1e-4)
    #         scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)  # eta_min 是最小学习率
    #         train_loader = make_Training_loader(config, tok, "{}_{}".format(i, j))
    #         train(config, model, tok, train_loader, optimizer, scheduler, layer=config.single_layer)
    #
    #         expert_layer = model.model.layers[11].mlp.moe.expert  # 定位到 Expert 层
    #         expert_params = expert_layer.state_dict()  # 获取参数
    #         torch.save(expert_params, os.path.join(config.save_path, '{}_{}.pth'.format(i,j)))
    #         recover_layer(model, config.single_layer, original_layer)

    # original_layer = model.model.layers[config.single_layer].mlp
    # replace_layer(model, config.single_layer, original_layer, config.num_experts)
    # optimizer = AdamW(model.parameters(), lr=1e-4)
    # scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)  # eta_min 是最小学习率
    # train_loader = make_Training_loader(config, tok, 'general_MOE')
    # train(config, model, tok, train_loader, optimizer, scheduler, layer=config.single_layer)
    #
    # expert_layer = model.model.layers[11].mlp.moe.expert  # 定位到 Expert 层
    # expert_params = expert_layer.state_dict()  # 获取参数
    # torch.save(expert_params, os.path.join(config.save_path, 'general_MOE.pth'))
    # recover_layer(model, config.single_layer, original_layer)

    # with open(config.valid_path) as f:
    #     datas = json.load(f)
    # for i in trange(1500):
    #     valid_loader = make_Validation_loader(config, tok, i)
    #
    #     ctxs = datas[i]["ctxs"][:config.num_experts]
    #     indexs = []
    #     for ctx in ctxs:
    #         if ctx['id'] in sub_dataset_expert:
    #             indexs.append(ctx['id'])
        # indexs, indexs_value = search_topK_with_softmax(query, rounter_model, K = config.num_experts)

        # if len(indexs) == 0:
        #     original_layer = model.model.layers[config.single_layer].mlp
        #     replace_layer(model, config.single_layer, original_layer, num_experts=1)
        #     updated_expert_params = {}
        #     expert_params = torch.load('/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ_MoE/general_MOE.pth')
        #     for key in expert_params:
        #         new_key = "model.layers.11.mlp.moe.experts.{0}." + key
        #         updated_expert_params[new_key] = expert_params[key]
        # else:
        #     num_experts = len(indexs)
        #     original_layer = model.model.layers[config.single_layer].mlp
        #     replace_layer(model, config.single_layer, original_layer, num_experts)
        #     updated_expert_params = {}
        #     for ie in range(num_experts):
        #         expert_params = torch.load('/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ_MoE/{}.pth'.format(indexs[ie]))
        #         for key in expert_params:
        #             new_key = "model.layers.11.mlp.moe.experts.{}.".format(ie) + key
        #             updated_expert_params[new_key] = expert_params[key]

        # model.load_state_dict(updated_expert_params, strict=False)
        # results = valid(model, tok, valid_loader, layer=config.single_layer)
        # recover_layer(model, config.single_layer, original_layer)

        # for key, value in zip(metrics.keys(), results):
        #     metrics[key].append(value)
        #
        # if (i+1) % 10 == 0:
        #     for key, value in metrics.items():
        #         print(key, len(metrics[key]), np.mean(metrics[key]) * 100)


if __name__ == '__main__':
    main()