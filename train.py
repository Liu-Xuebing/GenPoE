from tqdm import tqdm
from data.base import make_Training_loader
from utils import cross_entropy
from MOE_model.make_model import make_model, replace_layer
from MOE_model.hypernetwork import HyperNetwork
import hydra
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch



def create_pre_hook_fn(id, weights, delta):
    def pre_hook_fn(module, inputs):
        return (inputs[0], id, weights, delta)
    return pre_hook_fn



def train(config, hypernetwork, model, train_loader, optimizer, scheduler):
    hypernetwork.train()
    model.eval()
    running_loss = 0.0
    for ix, tuples in enumerate(tqdm(train_loader)):
        tok_tuples, tok_sentence = tuples
        if "Llama-2-7b-hf" in model.config._name_or_path:
            input_embeds = model.model.embed_tokens(tok_sentence[0]['input_ids']) # shape(batchsize, length, embedding_dim:4096)
            delta = hypernetwork(input_embeds)
            train_hook = model.model.layers[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0,[1], delta))
        elif "gpt-j-6B" in model.config._name_or_path:
            input_embeds = model.transformer.wte(tok_sentence[0]['input_ids']) # shape(batchsize, length, embedding_dim:4096)
            delta = hypernetwork(input_embeds)
            train_hook = model.transformer.h[config.single_layer].mlp.register_forward_pre_hook(create_pre_hook_fn(0,[1], delta))
        else:
            raise AssertionError
        wo_logits = model(**tok_tuples)["logits"]
        FT_loss = cross_entropy(wo_logits, tok_tuples["labels"])
        train_hook.remove()
        optimizer.zero_grad()
        FT_loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += FT_loss.item()
        if ix%200 == 199:
            print(f"Training Loss: {(running_loss/200):.4f}")
            running_loss = 0.0



@hydra.main(config_path="config", config_name="config")
def main(config):
    hypernetwork = HyperNetwork(4096, 512, 4096, 4096, 4096).cuda()
    model, tok = make_model(config)
    if "Llama-2-7b-hf" in model.config._name_or_path:
        original_layer = model.model.layers[config.single_layer].mlp
    elif "gpt-j-6B" in model.config._name_or_path:
        original_layer = model.transformer.h[config.single_layer].mlp
    else:
        raise AssertionError

    replace_layer(model, config.single_layer, original_layer, num_experts=config.num_experts)

    for name, param in model.named_parameters():
        param.requires_grad = False

    optimizer = AdamW(hypernetwork.parameters(), lr=config.learning_rate)
    train_loader = make_Training_loader(config, tok)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=config.learning_rate_min)
    train(config, hypernetwork, model, train_loader, optimizer, scheduler)

    torch.save(hypernetwork.state_dict(), config.hypernetwork_ckpt.format(config.model_name.split("/")[-1], config.data_name))


if __name__ == '__main__':
    main()