import os
from pathlib import Path
import random

from scipy.io.matlab.mio5_params import miDOUBLE
from tqdm import tqdm, trange
from data.base import make_Training_loader, make_Validation_loader
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import cross_entropy, cal_EM_F1, kl_divergence, vector_loss, sub_dataset_expert, tansfer_to_scftmax
from MOE_model.make_model import make_model, replace_layer, recover_layer
import hydra
from torch.optim import AdamW
import torch
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
import json
import numpy as np


# extracted_features = []
# def hook_fn(module, inputs):
#     extracted_features.append(inputs[0].detach())


post_feature=[]
def post_hook_fn(module, inputs, outputs):
    post_feature.append(outputs.detach())


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



# def train_router(config, model, tokenizer):
#     dataset = RouterDataset(config, config.train_router_path, tokenizer)
#     dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#     router = reranker().to(config.device_router)
#     optimizer = optim.AdamW(router.parameters(), lr=config.router_learning_rate)
#     loss_fn = RouterLoss(lambda_score = config.router_lambda_score).to(config.device_router)
#     scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=1e-6)
#
#     model.eval()  # 设为评估模式
#     router.train()
#     for step, batch in enumerate(tqdm(dataloader, desc="Training Progress", ncols=80)):
#         input_ids, score_indices = batch
#         labels = torch.zeros(config.neg_sample + 1, dtype=torch.float).to(config.device_router)
#         labels[0] = 1.0
#
#         input_ids = {key: value.squeeze(0) for key, value in input_ids.items()}
#         score_indices = [si[0] for si in score_indices]
#
#         hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(hook_fn)
#
#         extracted_features.clear()
#         with torch.no_grad():
#             model(**input_ids)
#
#         optimizer.zero_grad()
#
#         input_feature = extracted_features[0].to(config.device_router)
#         scores = router(input_feature, score_indices)
#
#         loss = loss_fn(scores, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         print(loss.item())
#
#         hook.remove()
#         if step % 200 == 0:
#             torch.save(router.state_dict(), "router_model.pth")



def train(config, original_model, model, train_loader, optimizer, scheduler):
    model.train()
    for epoch in range(config.epochs):
        running_losses = 0.0
        for tuples in train_loader:
            tok_tuples_w_passage, tok_tuples_wo_passage = tuples
            optimizer.zero_grad()

            # w_hook = original_model.model.layers[config.single_layer].mlp.register_forward_hook(post_hook_fn)
            # post_feature.clear()
            # w_logits = original_model(**tok_tuples_w_passage)["logits"]
            # w_M = post_feature[0]
            # w_hook.remove()
            #
            train_hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0,[1]))
            # wo_hook = model.model.layers[config.single_layer].mlp.register_forward_hook(post_hook_fn)
            # post_feature.clear()
            wo_logits = model(**tok_tuples_wo_passage)["logits"]
            # wo_M = post_feature[0]
            train_hook.remove()
            # wo_hook.remove()

            # V_loss = vector_loss(w_M, wo_M, tok_tuples_w_passage["labels"], tok_tuples_wo_passage["labels"])
            FT_loss = cross_entropy(wo_logits, tok_tuples_wo_passage["labels"])
            # KL_loss = kl_divergence(w_logits, wo_logits, tok_tuples_w_passage["labels"], tok_tuples_wo_passage["labels"])
            loss = FT_loss
            loss.backward()
            optimizer.step()
            running_losses += loss.item()
        scheduler.step()
        print(f"Training Loss: {running_losses:.4f}")


def valid_o(original_model, tok, valid_loader):
    original_model.eval()
    for tuples in tqdm(valid_loader, desc="Valid"):
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers = tuples
        base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, original_model, tok, answers)
        dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, original_model, tok, answers)
        pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, original_model, tok, answers)
    return base_EM, base_F1, dir_EM, dir_F1, pro_EM, pro_F1



def valid(config, model, tok, valid_loader, weights):
    model.eval()
    for tuples in tqdm(valid_loader, desc="Valid"):
        weights = tansfer_to_scftmax(weights)
        base_input, len_base_input, direct_input, len_direct_input, prompt_input, len_prompt_input, sentences, answers = tuples
        with torch.no_grad():
            hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0, weights))
            base_EM, base_F1 = EM_F1_of_model_generation(base_input, len_base_input, model, tok, answers)
            hook.remove()

            hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(direct_input.size(1)-base_input.size(1), weights))
            dir_EM, dir_F1 = EM_F1_of_model_generation(direct_input, len_direct_input, model, tok, answers)
            hook.remove()

            hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(prompt_input.size(1)-base_input.size(1), weights))
            pro_EM, pro_F1 = EM_F1_of_model_generation(prompt_input, len_prompt_input, model, tok, answers)
            hook.remove()

    return base_EM, base_F1, dir_EM, dir_F1, pro_EM, pro_F1


@hydra.main(config_path="config", config_name="config")
def main(config):
    if config.train_type == 'Train':
        model, tok = make_model(config)
        moe_model = copy.deepcopy(model)
        # train_files = os.listdir(config.train_path)
        train_files = []
        #-----------------------------------------------------------------------------------------------------
        with open('/data3/liuxb/datasets/NQ/NQ_test_rerank_results.json') as f:
            datas = json.load(f)
        for data in datas:
            ctx = data['ctxs'][:4]
            for c in ctx:
                train_files.append('{}.json'.format(c['id'][len('wiki:'):]))
        print(len(train_files))
        train_files = list(set(train_files))
        print(len(train_files))
        random.shuffle(train_files)
        #----------------------------------------------------------------------------------------------------

        for file in tqdm(train_files, desc='Train'):
            if os.path.exists(os.path.join(config.save_path, Path(file).with_suffix(".pth"))):
                continue
            original_layer = moe_model.model.layers[config.single_layer].mlp
            replace_layer(moe_model, config.single_layer, original_layer, num_experts=1)
            optimizer = AdamW(moe_model.parameters(), lr=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)  # eta_min 是最小学习率
            train_loader = make_Training_loader(config, tok, file)
            train(config, model, moe_model, train_loader, optimizer, scheduler)

            expert_layer = moe_model.model.layers[11].mlp.moes.experts[0]
            expert_params = expert_layer.state_dict()  # 获取参数
            torch.save(expert_params, os.path.join(config.save_path, Path(file).with_suffix(".pth")))

            recover_layer(moe_model, config.single_layer, original_layer)

    elif config.train_type == 'Test':
        model, tok = make_model(config)
        metrics = {
            "base_EM": [],
            "base_F1": [],
            "dir_EM": [],
            "dir_F1": [],
            "pro_EM": [],
            "pro_F1": []}
        with open(config.test_dataset_file) as f:
            total_datas = json.load(f)

        start_time = time.time()
        for ix, data in enumerate(total_datas[:100]):
            valid_loader = make_Validation_loader(config, tok, data)
            # ctxs = data["ctxs"][:config.number_experts]
            # weights = data['scores'][:config.number_experts]
            # has_indexs = []
            # for ctx in ctxs:
            #     if ctx['id'] in sub_dataset_expert:
            #         has_indexs.append(ctx['id'][len('wiki:'):])


            # assert len(has_indexs) == config.number_experts
            # original_layer = model.model.layers[config.single_layer].mlp
            # replace_layer(model, config.single_layer, original_layer, len(has_indexs))
            # updated_expert_params = {}
            # for ie in range(len(has_indexs)):
            #     expert_params = torch.load(os.path.join(config.save_path, "{}.pth".format(has_indexs[ie])))
            #     for key in expert_params:
            #         new_key = "model.layers.11.mlp.moes.experts.{}.".format(ie) + key
            #         updated_expert_params[new_key] = expert_params[key]
            # model.load_state_dict(updated_expert_params, strict=False)

            results = valid_o(model, tok, valid_loader)
            # recover_layer(model, config.single_layer, original_layer)


            # for key, value in zip(metrics.keys(), results):
            #     metrics[key].append(value)
            # #
            # if (ix+1) % 10 == 0:
            #     for key, value in metrics.items():
            #         print(key, len(metrics[key]), np.mean(metrics[key]) * 100)
        end_time = time.time()
        inference_time = end_time - start_time
        throughput = 100 / inference_time

        print(f"Throughput: {throughput:.3f} samples/second")


    else:
        raise Exception("Invalid train type")



if __name__ == '__main__':
    main()